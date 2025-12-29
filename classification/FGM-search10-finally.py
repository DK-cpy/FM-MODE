import os
import sys
import logging
import random
import torch.nn as nn
import genotypes
import utils
import torch.utils
import numpy as np
from torch.autograd import Variable
from model_search import Network
import argparse
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from population import Population
from functools import partial
import pickle
from diffevo import DiffEvo
import torch
from torch.distributions import Bernoulli
from thop import profile

# 全局变量
best_accuracy = -np.inf
best_model_path = 'best_teacher.pth'
CIFAR_CLASSES = 10


# 适应度计算函数（基于验证集损失）
def fitness_function(model, inputs, targets, device):
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        prec1, _ = utils.accuracy(logits, targets, topk=(1, 5))
        return -loss.item(), prec1.item()  # 负损失（作为适应度）和准确率


# 计算模型参数和计算量
def compute_params_and_flops(model, genotype, device, args):
    from model import NetworkCIFAR
    model_new = NetworkCIFAR(
        C=args.init_channels,
        num_classes=CIFAR_CLASSES,
        layers=args.layers,
        auxiliary=False,
        genotype=genotype
    )
    model_new.to(device)
    model_new.drop_path_prob = 0.0
    model_new.eval()

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        macs, params = profile(model_new, inputs=(dummy_input,), verbose=False)
        madds = macs * 2
        logging.debug(f"模型参数: Params={params / 1e6:.2f}M, MAdds={madds / 1e6:.2f}M")
        return params, madds


# NSGA-II相关函数
def non_dominated_sorting(objectives):
    n = len(objectives)
    S = [[] for _ in range(n)]
    nP = [0] * n
    rank = [0] * n
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dominates = True
            for k in range(len(objectives[i])):
                if objectives[i][k] > objectives[j][k]:
                    dominates = False
                    break
            if dominates:
                S[i].append(j)
            elif all(objectives[j][k] <= objectives[i][k] for k in range(len(objectives[j]))):
                nP[i] += 1
        if nP[i] == 0:
            rank[i] = 0
            fronts[0].append(i)

    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                nP[q] -= 1
                if nP[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)

    return fronts[:-1]


# ==================== 替换为imagenet-doubao.py中的改进_crowding_distance方法 ====================
def crowding_distance(objectives, front):
    """
    改进点：
    1. 用四分位距（IQR）替代max-min标准化，抗异常值；
    2. 增加局部拥挤度（k近邻），平衡全局与局部多样性；
    3. 保留原函数参数接口（仅objectives和front），不影响其他调用。
    """
    n = len(front)
    if n == 0:
        return []
    if n == 1:
        return [float('inf')]  # 单个个体拥挤度无穷大，确保保留

    # 1. 处理目标值异常（替换NaN/inf）
    objectives_array = np.array(objectives, dtype=np.float32)
    objectives_array = np.nan_to_num(objectives_array, nan=1e18, posinf=1e18, neginf=-1e18)
    m = objectives_array.shape[1] if objectives_array.ndim > 1 else 1

    # 2. 初始化拥挤度数组
    crowd_dist = np.zeros(n)

    # 3. 计算全局拥挤度（IQR标准化，抗异常值）
    for obj_idx in range(m):
        # 提取当前前沿在该目标上的取值
        obj_vals = objectives_array[front, obj_idx] if m > 1 else objectives_array[front]
        # 用四分位距（IQR）避免异常值影响
        q1 = np.percentile(obj_vals, 25)
        q3 = np.percentile(obj_vals, 75)
        iqr = q3 - q1
        if iqr < 1e-8:  # 目标值无差异，跳过
            continue
        # 对目标值排序，获取排序索引
        sorted_idx = np.argsort(obj_vals)
        # 首尾个体全局拥挤度设为无穷大（保留边界多样性）
        crowd_dist[sorted_idx[0]] = float('inf')
        crowd_dist[sorted_idx[-1]] = float('inf')
        # 计算中间个体的全局拥挤度
        for i in range(1, n-1):
            prev_val = obj_vals[sorted_idx[i-1]]
            next_val = obj_vals[sorted_idx[i+1]]
            # 累加拥挤度（IQR标准化）
            crowd_dist[sorted_idx[i]] += np.abs(next_val - prev_val) / iqr

    # 4. 计算局部拥挤度（k近邻密度，避免局部拥挤）
    k_neighbor = 5  # 近邻数量，经验值
    front_obj = objectives_array[front]  # 前沿个体的目标矩阵
    for i in range(n):
        # 计算当前个体与其他个体的欧氏距离
        dists = np.linalg.norm(front_obj - front_obj[i], axis=1) if m > 1 else np.abs(front_obj - front_obj[i])
        # 取k个最近邻（排除自身）
        k = min(k_neighbor, len(dists)-1)
        if k <= 0:
            local_d = 0.0
        else:
            k_dist = np.sort(dists)[1:k+1]  # 第0个是自身（距离0）
            local_d = 1.0 / (np.mean(k_dist) + 1e-8)  # 局部越稀疏，值越大
        # 融合全局与局部拥挤度（7:3权重）
        crowd_dist[i] = 0.7 * crowd_dist[i] + 0.3 * local_d

    # 5. 处理无穷大（避免后续计算异常）
    crowd_dist[np.isinf(crowd_dist)] = 1e18
    return crowd_dist.tolist()


# ==================== 替换为imagenet-doubao.py中的改进_nsga2_sa_selection方法（适配独立函数） ====================
def nsga2_sa_selection(population, objectives, pop_size, sa_initial_temp=100.0, sa_current_temp=100.0):
    """
    改进点：
    1. 融合拥挤度到模拟退火能量计算，优先保留"优质+多样"个体；
    2. 增加精英保留机制，强制保留前5%优质个体；
    3. 优化最后一批前沿补充逻辑，不再随机选择；
    4. 适配为独立函数，添加模拟退火参数。
    """
    # 1. 预处理目标函数值
    if isinstance(objectives, np.ndarray):
        objectives_list = objectives.tolist()
    else:
        objectives_list = objectives
    total_ind = len(objectives_list)
    if total_ind == 0:
        raise ValueError("待选择种群为空")

    # 2. 非支配排序
    fronts = non_dominated_sorting(objectives_list)
    valid_fronts = []
    for front in fronts:
        valid_front = [int(idx) for idx in front if isinstance(idx, (int, np.integer)) and 0 <= idx < total_ind]
        if valid_front:
            valid_fronts.append(valid_front)
    if not valid_fronts:
        # 兜底：随机选择
        return [population[i] for i in np.random.choice(total_ind, pop_size, replace=True).tolist()]

    # 3. 精英保留（强制保留前5%优质个体）
    elite_ratio = 0.05
    elite_num = max(1, int(pop_size * elite_ratio))
    elite_candidates = []
    for front in valid_fronts:
        if len(elite_candidates) >= elite_num:
            break
        # 用改进后的拥挤度计算
        crowd_dist = crowding_distance(objectives_list, front)
        # 按"非支配等级+拥挤度"排序，取前k个
        front_with_dist = list(zip(front, crowd_dist))
        front_with_dist.sort(key=lambda x: -x[1])  # 拥挤度越高越优先
        elite_candidates.extend([idx for idx, _ in front_with_dist])
    # 去重并截取精英
    elite_set = list(dict.fromkeys(elite_candidates))[:elite_num]
    selected_indices = elite_set.copy()
    remaining = pop_size - len(selected_indices)

    # 4. 选择剩余个体（融合拥挤度的模拟退火）
    for front in valid_fronts:
        if remaining <= 0:
            break
        # 排除已选精英
        front_non_elite = [idx for idx in front if idx not in selected_indices]
        if not front_non_elite:
            continue

        # 若当前前沿可全部加入，直接选择
        if len(selected_indices) + len(front_non_elite) <= pop_size:
            selected_indices.extend(front_non_elite)
            remaining -= len(front_non_elite)
            continue

        # 5. 模拟退火选择（融合拥挤度）
        # 5.1 计算拥挤度
        crowd_dist = crowding_distance(objectives_list, front_non_elite)
        # 5.2 归一化目标值（计算能量）
        front_fitness = np.array([objectives_list[idx] for idx in front_non_elite])
        obj_min = front_fitness.min(axis=0)
        obj_max = front_fitness.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range < 1e-8] = 1e-8
        normalized_obj = (front_fitness - obj_min) / obj_range  # 越小越好

        # 5.3 融合拥挤度的能量计算（能量=目标和-拥挤度权重×拥挤度）
        sa_temp = sa_current_temp
        crowd_weight = 0.1 + 0.4 * (sa_temp / sa_initial_temp)  # 动态权重：0.1~0.5
        crowd_weight = np.clip(crowd_weight, 0.1, 0.5)
        # 能量越低，个体越优（目标优+多样性好）
        if normalized_obj.ndim > 1:
            obj_sum = normalized_obj.sum(axis=1)
        else:
            obj_sum = normalized_obj
        energy = obj_sum - crowd_weight * np.array(crowd_dist)

        # 5.4 退火选择
        sa_candidates = list(zip(front_non_elite, energy))
        selected_from_front = []
        while remaining > 0 and sa_candidates:
            if sa_temp < 1e-4:
                # 低温：纯贪心（选能量最低）
                sa_candidates.sort(key=lambda x: x[1])
                chosen = sa_candidates.pop(0)
            else:
                # 高温：概率选择（能量低的概率大）
                min_energy = min(c[1] for c in sa_candidates)
                rel_energy = [c[1] - min_energy for c in sa_candidates]
                probs = np.exp(-np.array(rel_energy) / sa_temp)
                probs = probs / probs.sum()
                idx = np.random.choice(len(sa_candidates), p=probs)
                chosen = sa_candidates.pop(idx)
            selected_from_front.append(chosen[0])
            remaining -= 1
        selected_indices.extend(selected_from_front)

    # 6. 兜底补充（若仍不足，选拥挤度高的个体）
    if len(selected_indices) < pop_size:
        # 收集所有未选个体
        unselected = [idx for idx in range(total_ind) if idx not in selected_indices]
        if unselected:
            # 计算未选个体的拥挤度
            crowd_dist = crowding_distance(objectives_list, unselected)
            unselected_with_dist = list(zip(unselected, crowd_dist))
            unselected_with_dist.sort(key=lambda x: -x[1])  # 拥挤度高优先
            supplement = [idx for idx, _ in unselected_with_dist[:pop_size - len(selected_indices)]]
            selected_indices.extend(supplement)
        else:
            # 极端情况：随机补充
            selected_indices.extend(np.random.choice(total_ind, pop_size - len(selected_indices), replace=True).tolist())

    # 7. 最终校验（去重+索引有效）
    selected_indices = list(dict.fromkeys(selected_indices))  # 去重
    selected_indices = [int(idx) for idx in selected_indices if 0 <= idx < len(population)]
    # 确保种群大小
    while len(selected_indices) < pop_size:
        selected_indices.append(np.random.choice(selected_indices))

    return [population[i] for i in selected_indices[:pop_size]]


# 参数格式转换工具
def convert_to_two_element_list(arch_params, device):
    if isinstance(arch_params, list) and len(arch_params) == 2:
        return [torch.as_tensor(p, device=device) for p in arch_params]
    elif isinstance(arch_params, torch.Tensor):
        if arch_params.shape == (28, 8):
            alphas_normal = arch_params[:14, :].clone()
            alphas_reduce = arch_params[14:, :].clone()
            return [alphas_normal, alphas_reduce]
        elif arch_params.shape == (14, 8):
            return [arch_params.clone(), arch_params.clone()]
        elif arch_params.dim() == 1 and arch_params.numel() == 224:
            arch_params_2d = arch_params.view(28, 8)
            alphas_normal = arch_params_2d[:14, :]
            alphas_reduce = arch_params_2d[14:, :]
            return [alphas_normal, alphas_reduce]
        else:
            raise ValueError(f"不支持的张量形状: {arch_params.shape}，需为(28,8)或(14,8)")
    elif isinstance(arch_params, np.ndarray):
        arch_params_tensor = torch.as_tensor(arch_params, device=device)
        return convert_to_two_element_list(arch_params_tensor, device)
    else:
        raise ValueError(f"不支持的参数类型: {type(arch_params)}")

'''
# 适应度评估函数（修复核心错误）
def fitness_function_wrapper(x_population, model, valid_queue, criterion, device, max_batches=2):
    model.eval()
    fitness_values = []

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            if step >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_fitness_values = []

            for i, arch_parameters in enumerate(x_population):
                # 确保输入是Tensor
                if isinstance(arch_parameters, list):
                    combined = torch.cat(arch_parameters, dim=0) if len(arch_parameters) == 2 else torch.tensor(
                        arch_parameters, device=device)
                elif isinstance(arch_parameters, np.ndarray):
                    combined = torch.tensor(arch_parameters, device=device, dtype=torch.float32)
                else:
                    combined = arch_parameters.to(device) if isinstance(arch_parameters,
                                                                        torch.Tensor) else torch.tensor(arch_parameters,
                                                                                                        device=device)

                # 处理参数形状（224元素对应28x8，拆分为两个14x8）
                if combined.nelement() != 224:
                    reshaped = False
                    for dim_size in combined.shape:
                        if dim_size == 224:
                            combined = combined.view(-1, 224)[0]
                            reshaped = True
                            break
                    if not reshaped:
                        logging.error(f"无法重塑参数 {i}，当前形状: {combined.size()}")
                        continue
                combined = combined.view(28, 8)  # 224 = 28*8
                alphas_normal = combined[:14, :]
                alphas_reduce = combined[14:, :]

                model.copy_arch_parameters([alphas_normal, alphas_reduce])
                logits = model(inputs)
                loss = criterion(logits, targets)
                batch_fitness_values.append(-loss.item())

            if batch_fitness_values:
                fitness_values.append(batch_fitness_values)

        if fitness_values:
            return torch.tensor([sum(f) / max_batches for f in zip(*fitness_values)], dtype=torch.float32)
        else:
            logging.warning("未计算到有效适应度值")
            return torch.tensor([], dtype=torch.float32)
'''
def fitness_function_wrapper(x_population, model, valid_queue, criterion, device, max_batches=2):
    model.eval()
    fitness_values = []

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            if step >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_fitness_values = []

            for i, individual in enumerate(x_population):
                # 获取架构参数
                arch_parameters = individual.arch_parameters if hasattr(individual, 'arch_parameters') else individual
                
                # 确保输入是Tensor
                if isinstance(arch_parameters, list):
                    # 如果是列表，假设包含两个张量 [alphas_normal, alphas_reduce]
                    if len(arch_parameters) == 2:
                        combined = torch.cat([p.flatten() for p in arch_parameters])
                    else:
                        combined = torch.tensor(arch_parameters, device=device, dtype=torch.float32)
                elif isinstance(arch_parameters, np.ndarray):
                    combined = torch.tensor(arch_parameters, device=device, dtype=torch.float32)
                elif isinstance(arch_parameters, torch.Tensor):
                    combined = arch_parameters.to(device)
                else:
                    # 尝试直接转换
                    try:
                        combined = torch.tensor(arch_parameters, device=device, dtype=torch.float32)
                    except:
                        logging.error(f"无法转换参数 {i}，类型: {type(arch_parameters)}")
                        continue

                # 处理参数形状（224元素对应28x8，拆分为两个14x8）
                if combined.nelement() != 224:
                    # 尝试重塑
                    try:
                        if combined.nelement() == 112:  # 只有一半参数
                            combined = torch.cat([combined, combined])  # 复制一份
                        elif combined.nelement() > 224:
                            combined = combined[:224]  # 截取前224个元素
                        else:
                            # 填充到224
                            padding = torch.zeros(224 - combined.nelement(), device=device)
                            combined = torch.cat([combined, padding])
                    except Exception as e:
                        logging.error(f"无法重塑参数 {i}，当前形状: {combined.size()}，错误: {str(e)}")
                        continue
                
                try:
                    combined = combined.view(28, 8)  # 224 = 28*8
                    alphas_normal = combined[:14, :]
                    alphas_reduce = combined[14:, :]

                    model.copy_arch_parameters([alphas_normal, alphas_reduce])
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                    batch_fitness_values.append(-loss.item())
                except Exception as e:
                    logging.error(f"模型前向传播失败 {i}，错误: {str(e)}")
                    batch_fitness_values.append(-10.0)  # 默认低适应度

            if batch_fitness_values:
                fitness_values.append(batch_fitness_values)

        if fitness_values:
            return torch.tensor([sum(f) / max_batches for f in zip(*fitness_values)], dtype=torch.float32)
        else:
            logging.warning("未计算到有效适应度值")
            return torch.tensor([-10.0] * len(x_population), dtype=torch.float32)  # 返回默认适应度
# 训练相关函数
def warm_train(model, train_queue, criterion, optimizer, device):
    model.train()
    for step, (inputs, targets) in enumerate(train_queue):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()


def train(model, train_queue, criterion, optimizer, gen, population, device, args):
    model.train()
    for step, (inputs, targets) in enumerate(train_queue):
        current_ind = population.get_population()[step % args.pop_size]
        current_alphas = convert_to_two_element_list(current_ind.arch_parameters, device)
        model.copy_arch_parameters(current_alphas)

        n = inputs.size(0)
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        current_ind.objs.update(loss.data, n)
        current_ind.top1.update(prec1.data, n)
        current_ind.top5.update(prec5.data, n)

        if (step + 1) % 100 == 0:
            loss_avg = current_ind.objs.avg.item() if isinstance(current_ind.objs.avg,
                                                                 torch.Tensor) else current_ind.objs.avg
            top1_avg = current_ind.top1.avg.item() if isinstance(current_ind.top1.avg,
                                                                 torch.Tensor) else current_ind.top1.avg
            top5_avg = current_ind.top5.avg.item() if isinstance(current_ind.top5.avg,
                                                                 torch.Tensor) else current_ind.top5.avg

            logging.info(
                f"[{gen} Generation] Batch #{step} 个体 {step % args.pop_size}/{len(population.get_population())} "
                f"损失: {loss_avg:.4f}, 准确率: {top1_avg:.2f}%")


# 验证函数
def validation_fitness(model, valid_queue, criterion, gen, population, device, args):
    model.eval()
    val_batch = next(iter(valid_queue))
    val_inputs, val_targets = val_batch[0][:args.fitness_batch_size], val_batch[1][:args.fitness_batch_size]
    objectives = []

    for i in range(len(population.get_population())):
        individual = population.get_population()[i]
        individual.objs.reset()
        individual.top1.reset()
        individual.top5.reset()

        try:
            arch_params_two_list = convert_to_two_element_list(individual.arch_parameters, device)
            discrete_alphas = utils.discretize(arch_params_two_list, device)
        except Exception as e:
            logging.error(f"个体 {i + 1} 参数转换失败: {str(e)}，使用默认参数")
            discrete_alphas = [torch.zeros(14, 8, device=device), torch.zeros(14, 8, device=device)]

        model.copy_arch_parameters(discrete_alphas)
        genotype = model.genotype()
        fitness_score, prec1 = fitness_function(model, val_inputs, val_targets, device)
        params, madds = compute_params_and_flops(model, genotype, device, args)

        individual.fitness_score = fitness_score
        individual.accuracy = prec1
        individual.params = params
        individual.madds = madds
        individual.arch_parameters = discrete_alphas
        objectives.append([-fitness_score, params, madds])

        shape_info = [tuple(p.shape) for p in discrete_alphas] if isinstance(discrete_alphas, list) else tuple(
            discrete_alphas.shape)
        logging.info(f"[{gen} Generation] 个体 {i + 1}/{len(population.get_population())} "
                     f"适应度: {fitness_score:.4f}, 准确率: {prec1:.2f}%, "
                     f"参数: {params / 1e6:.2f}M, 计算量: {madds / 1e6:.2f}M, 形状: {shape_info}")

    # NSGA-II选择（使用改进后的方法）
    pop_list = population.get_population()
    # 设置模拟退火参数（可根据需要调整）
    sa_initial_temp = 100.0
    sa_current_temp = max(50.0, sa_initial_temp * (0.95 ** gen))  # 随代数衰减
    
    selected_pop = nsga2_sa_selection(
        pop_list, 
        objectives, 
        args.pop_size,
        sa_initial_temp=sa_initial_temp,
        sa_current_temp=sa_current_temp
    )

    # 更新种群
    if hasattr(population, 'set_population'):
        population.set_population(selected_pop)
    elif hasattr(population, '_population'):
        population._population = selected_pop
    else:
        logging.warning("创建新种群对象")
        new_population = Population(args.pop_size, model._steps, device)
        for i, individual in enumerate(selected_pop[:len(new_population.get_population())]):
            new_ind_alphas = convert_to_two_element_list(individual.arch_parameters, device)
            individual.arch_parameters = new_ind_alphas
            new_population.get_population()[i] = individual
        population = new_population

    # 记录最佳个体
    best_individual = max(population.get_population(), key=lambda x: x.fitness_score)
    try:
        best_alphas_two_list = convert_to_two_element_list(best_individual.arch_parameters, device)
        best_alphas = utils.discretize(best_alphas_two_list, device)
    except Exception as e:
        logging.error(f"最佳个体参数转换失败: {str(e)}")
        best_alphas = [torch.zeros(14, 8, device=device), torch.zeros(14, 8, device=device)]

    model.copy_arch_parameters(best_alphas)
    best_genotype = model.genotype()
    logging.info(f"[{gen} Generation] 最佳基因型: {best_genotype}")
    return population


# 引入新个体函数
def introduce_new_individuals(population, num_new_individuals, device):
    indices = random.sample(range(len(population.get_population())), num_new_individuals)
    for idx in indices:
        new_individual = population.create_new_individual()
        new_individual.arch_parameters = convert_to_two_element_list(new_individual.arch_parameters, device)
        new_individual.fitness_score = -np.inf
        new_individual.accuracy = 0.0
        population.get_population()[idx] = new_individual
    logging.info(f"引入 {num_new_individuals} 个新个体")


# 模拟退火参数调整
def adjust_sa_parameters(epoch, total_epochs):
    initial_temperature = 100.0 * (1 - epoch / total_epochs)
    final_temperature = 0.1
    cooling_rate = 0.95 - 0.05 * (epoch / total_epochs)
    return initial_temperature, final_temperature, cooling_rate


# 主程序
def main():
    global args, device, best_accuracy
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data', type=str, default='fasterdatasets/cifar-10', help='数据路径')
    parser.add_argument('--dir', type=str, default=None, help='实验目录')
    parser.add_argument('--cutout', action='store_true', default=False, help='使用cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout长度')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮次')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备号')
    parser.add_argument('--tsize', type=int, default=10, help='锦标赛大小')
    parser.add_argument('--num_elites', type=int, default=50, help='精英个体数')
    parser.add_argument('--mutate_rate', type=float, default=0.2, help='变异率')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='初始学习率')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='最小学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=5, help='梯度裁剪')
    parser.add_argument('--train_portion', type=float, default=0.5, help='训练数据比例')
    parser.add_argument('--pop_size', type=int, default=50, help='种群大小')
    parser.add_argument('--report_freq', type=float, default=50, help='报告频率')
    parser.add_argument('--init_channels', type=int, default=16, help='初始通道数')
    parser.add_argument('--layers', type=int, default=8, help='层数')
    parser.add_argument('--warm_up', type=int, default=0, help='预热轮次')
    parser.add_argument('--fitness_batch_size', type=int, default=128, help='适应度评估批次大小')
    parser.add_argument('--lambda_fm', type=float, default=1.0, help='FGM损失权重')
    args = parser.parse_args()

    # 初始化随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 设备配置
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)
    cudnn.deterministic = True
    cudnn.enabled = True
    cudnn.benchmark = False

    # 模型和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, device)
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min
    )

    # 数据加载
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, pin_memory=False, num_workers=2,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, pin_memory=False, num_workers=2,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train])
    )

    # 初始化种群
    population = Population(args.pop_size, model._steps, device)
    for individual in population.get_population():
        individual.fitness_score = -np.inf
        individual.accuracy = 0.0
        individual.params = float('inf')
        individual.madds = float('inf')
        individual.arch_parameters = convert_to_two_element_list(individual.arch_parameters, device)

    # 目录和日志配置
    DIR = f"FGM-multi-objective-diffevo-{time.strftime('%Y%m%d-%H%M%S')}"
    if args.dir:
        DIR = os.path.join(args.dir, DIR)
    else:
        DIR = os.path.join(os.getcwd(), DIR)
    utils.create_exp_dir(DIR)
    utils.create_exp_dir(os.path.join(DIR, "weights"))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(os.path.join(DIR, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(f"GPU设备: {args.gpu}")
    logging.info(f"参数: {args}")

    # TensorBoard
    writer = SummaryWriter(os.path.join(DIR, 'runs'))

    # 初始化教师模型
    teacher_model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, device).to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 预热训练
    start = time.time()
    if args.warm_up > 0:
        for epoch in range(args.warm_up):
            start_time = time.time()
            logging.info('[INFO] 预热训练...')
            warm_train(model, train_queue, criterion, optimizer, device)
            logging.info(f"预热完成，耗时 {(time.time() - start_time) / 60:.2f} 分钟")

    # 初始化差分进化优化器
    de_optimizer = DiffEvo(
        teacher_model=teacher_model,
        num_step=1,
        density='kde',
        noise=1.0,
        lambda_kl=0.5,
        lambda_ce=1.0,
        lambda_fm=args.lambda_fm,
        #problem=None,  
        perturb_scale=0.1,
        kde_bandwidth=0.1
    )
    density_method = de_optimizer.get_density_method()
    logging.info(f"密度估计算法: {density_method}")
    writer.add_text('密度估计', density_method)

    # 主训练循环
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch} 学习率: {scheduler.get_lr()[0]}")

        # 引入新个体
        if epoch % 5 == 0:
            introduce_new_individuals(population, 5, device)

        # 训练种群
        start_time = time.time()
        logging.info(f"第 {epoch + 1} 代训练开始")
        train(model, train_queue, criterion, optimizer, epoch + 1, population, device, args)
        logging.info(f"训练完成，耗时 {(time.time() - start_time) / 60:.2f} 分钟")

        # 保存模型
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in model_state.items() if 'total_ops' not in k and 'total_params' not in k}
        torch.save(filtered_state, os.path.join(DIR, "model.pt"))
        scheduler.step()

        # 适应度验证
        logging.info(f"第 {epoch + 1} 代适应度评估")
        population = validation_fitness(model, valid_queue, criterion, epoch + 1, population, device, args)

        # 差分进化优化
        logging.info("使用FGM优化种群")
        arch_params_list = []
        for ind in population.get_population():
            if isinstance(ind.arch_parameters, list) and len(ind.arch_parameters) == 2:
                params_flat = np.concatenate([p.cpu().numpy().flatten() for p in ind.arch_parameters])
            elif isinstance(ind.arch_parameters, torch.Tensor):
                params_flat = ind.arch_parameters.cpu().numpy().flatten()
            else:
                params_flat = np.array(ind.arch_parameters).flatten()
            arch_params_list.append(params_flat)
        arch_params_np = np.array(arch_params_list)

        # 准备训练数据
        train_data_batch = next(iter(train_queue))
        train_data_fgm = (train_data_batch[0].to(device), train_data_batch[1].to(device))
        fitness_fn = partial(fitness_function_wrapper, model=model, valid_queue=valid_queue,
                     criterion=criterion, device=device)
        optimized_params, _ = de_optimizer.optimize(initial_population=arch_params_np,
                                                    train_data=train_data_fgm,
                                                    valid_queue=valid_queue,
                                                    fitness_fn=fitness_fn, 
                                                    
                                                    # 使用新参数名
                                                    trace=True)
        '''
        fitness_fn = partial(fitness_function_wrapper, model=model, valid_queue=valid_queue,
                             criterion=criterion, device=device)
        optimized_params, _ = de_optimizer.optimize(
            initial_population=arch_params_np,
            train_data=train_data_fgm,
            valid_queue=valid_queue,
            fitness_fn=fitness_fn,
            trace=True
        )
'''
        # 更新种群参数
        optimized_params_np = optimized_params.cpu().numpy() if isinstance(optimized_params,
                                                                           torch.Tensor) else optimized_params
        for i, (ind, params) in enumerate(zip(population.get_population(), optimized_params_np)):
            ind.arch_parameters = convert_to_two_element_list(params, device)

        # 更新最佳模型
        current_best = max(population.get_population(), key=lambda x: x.fitness_score)
        if current_best.fitness_score > best_accuracy:
            best_accuracy = current_best.fitness_score
            best_state = model.state_dict()
            best_filtered = {k: v for k, v in best_state.items() if 'total_ops' not in k and 'total_params' not in k}
            torch.save(best_filtered, best_model_path)
            logging.info(f"保存最佳模型，适应度: {best_accuracy:.4f}, 准确率: {current_best.accuracy:.2f}%")
            teacher_model.load_state_dict(best_filtered)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            de_optimizer.teacher_model = teacher_model

        # 记录指标
        for i, p in enumerate(population.get_population()):
            writer.add_scalar(f"种群/适应度_{i + 1}", p.fitness_score, epoch + 1)
            writer.add_scalar(f"种群/准确率_{i + 1}", p.accuracy, epoch + 1)

        # 保存种群
        with open(os.path.join(DIR, f"population_{epoch + 1}.pickle"), 'wb') as f:
            pickle.dump(population, f)

        logging.info(f"第 {epoch + 1}/{args.epochs} 代完成，耗时 {(time.time() - start_time) / 60:.2f} 分钟")

    # 总耗时
    total_time = (time.time() - start) / 3600
    logging.info(f"总搜索时间: {total_time:.2f} 小时")
    writer.close()


if __name__ == '__main__':
    import multiprocessing as mp

    mp.freeze_support()
    main()