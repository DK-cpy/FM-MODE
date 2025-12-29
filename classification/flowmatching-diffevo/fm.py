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
from diffevo import DiffEvo  # 导入修改后的DiffEvo
import torch
from torch.distributions import Bernoulli
from thop import profile


# -------------------------- 全局工具函数（新增：张量标准化） --------------------------
def standardize_arch_params(params, device):
    """
    标准化架构参数为1维标量张量，处理所有可能的异常类型：
    - 列表（可能包含标量或张量）
    - 多维张量
    - numpy数组（可能为多维）
    """
    processed = []
    
    # 处理列表类型
    if isinstance(params, list):
        for p in params:
            # 若列表元素是张量
            if isinstance(p, torch.Tensor):
                # 确保张量是标量（1个元素）
                if p.numel() == 1:
                    processed.append(p.item())  # 提取标量值
                else:
                    # 多维张量取均值（强制转为标量）
                    processed.append(p.mean().item())
            # 若列表元素是numpy数组
            elif isinstance(p, np.ndarray):
                if p.size == 1:
                    processed.append(p.item())
                else:
                    processed.append(p.mean().item())
            # 其他类型（int/float）
            else:
                processed.append(float(p))
    
    # 处理张量类型
    elif isinstance(params, torch.Tensor):
        # 展平为1维
        flat_params = params.flatten()
        # 逐个元素检查（确保都是标量）
        for p in flat_params:
            processed.append(p.item())
    
    # 处理numpy数组
    elif isinstance(params, np.ndarray):
        # 展平为1维
        flat_params = params.flatten()
        for p in flat_params:
            processed.append(float(p))
    
    # 其他类型（如单个数值）
    else:
        processed.append(float(params))
    
    # 转换为1维张量并返回
    return torch.tensor(processed, device=device, dtype=torch.float32).flatten()


# -------------------------- 全局函数（NASWOT/NSGA-II） --------------------------
def naswot_sort_key(individual):
    return -individual.naswot_score

best_accuracy = -np.inf
best_model_path = 'best_teacher.pth'

def compute_naswot_score(model, inputs, device):
    """计算未训练模型的NASWOT分数，添加详细调试日志"""
    model.eval()
    with torch.no_grad():
        activations = []
        hooks = []
        relu_count = 0  # 统计注册的ReLU层数量
        
        # 注册钩子捕获ReLU激活
        def hook_fn(module, input, output):
            nonlocal relu_count
            if isinstance(module, nn.ReLU):
                relu_count += 1  # 记录实际触发的ReLU层
                binary_act = (output > 0).float()
                batch_size = binary_act.size(0)
                flattened = binary_act.view(batch_size, -1)
                activations.append(flattened)
                # 调试：打印激活形状
                logging.debug(f"捕获ReLU激活: 批次大小={batch_size}, 激活维度={flattened.shape[1]}")
        
        # 统计模型中所有ReLU层数量
        total_relu_in_model = sum(1 for module in model.modules() if isinstance(module, nn.ReLU))
        logging.debug(f"模型中ReLU层总数: {total_relu_in_model}")
        
        # 注册所有ReLU层的钩子
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # 前向传播获取激活
        model(inputs.to(device))
        logging.debug(f"前向传播后实际捕获的ReLU激活数: {relu_count}")
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 检查是否捕获到激活
        if not activations:
            logging.warning("未捕获到任何ReLU激活！可能原因：模型无ReLU层或输入未触发激活")
            return -np.inf
        
        # 检查激活维度是否合理
        all_activations = torch.cat(activations, dim=1)
        num_activation_units = all_activations.size(1)
        batch_size = all_activations.size(0)
        logging.debug(f"合并后激活维度: {all_activations.shape} (批次={batch_size}, 特征数={num_activation_units})")
        
        # 检查激活是否全为0（未触发任何激活）
        if torch.all(all_activations == 0):
            logging.warning("所有ReLU激活均为0！输入数据可能未触发激活")
            return -np.inf
        
        # 计算汉明距离核矩阵 K_H
        try:
            hamming_dist = torch.cdist(all_activations, all_activations, p=0)
            K_H = num_activation_units - hamming_dist
            K_H += torch.eye(batch_size, device=device) * 1e-6  # 避免奇异矩阵
        except Exception as e:
            logging.error(f"核矩阵计算失败: {str(e)}")
            return -np.inf
        
        # 计算核矩阵行列式的对数
        try:
            log_det = torch.logdet(K_H)
            logging.debug(f"NASWOT计算成功: log_det={log_det.item()}")
            return log_det.item()
        except RuntimeError as e:
            logging.warning(f"核矩阵奇异，无法计算行列式: {str(e)}")
            return -np.inf
    

CIFAR_CLASSES = 10 

def compute_params_and_flops(model, genotype, device):
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
        return params, madds

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

def crowding_distance(objectives, front):
    n = len(front)
    if n == 0:
        return []
    distances = [0.0] * n
    num_objectives = len(objectives[0])
    for m in range(num_objectives):
        sorted_front = sorted(front, key=lambda i: objectives[i][m])
        distances[0] = float('inf')
        distances[-1] = float('inf')
        min_val = objectives[sorted_front[0]][m]
        max_val = objectives[sorted_front[-1]][m]
        if max_val - min_val < 1e-6:
            continue
        for i in range(1, n-1):
            idx = sorted_front[i]
            next_val = objectives[sorted_front[i+1]][m]
            prev_val = objectives[sorted_front[i-1]][m]
            distances[i] += (next_val - prev_val) / (max_val - min_val)
    return distances

def nsga2_selection(population, objectives, pop_size):
    fronts = non_dominated_sorting(objectives)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            distances = crowding_distance(objectives, front)
            sorted_front = sorted(zip(front, distances), key=lambda x: -x[1])
            remaining = pop_size - len(selected)
            selected.extend([idx for idx, _ in sorted_front[:remaining]])
            break
    return [population[i] for i in selected]

# -------------------------- 解析参数 --------------------------
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='fasterdatasets/cifar-10', help='location of the data corpus')
parser.add_argument('--dir', type=str, default=None, help='location of trials')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--tsize', type=int, default=10, help='Tournament size')
parser.add_argument('--num_elites', type=int, default=50, help='Number of Elites')
parser.add_argument('--mutate_rate', type=float, default=0.2, help='mutation rate')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--pop_size', type=int, default=50, help='population size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--warm_up', type=int, default=0, help='warm up epochs')
parser.add_argument('--naswot_batch_size', type=int, default=128, help='batch size for NASWOT evaluation')
parser.add_argument('--lambda_fm', type=float, default=1.0, help='FGM loss weight')
args = parser.parse_args()

# -------------------------- 训练函数 --------------------------
def warm_train(model, train_queue, criterion, optimizer):
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

def train(model, train_queue, criterion, optimizer, gen):
    model.train()
    for step, (inputs, targets) in enumerate(train_queue):
        model.copy_arch_parameters(population.get_population()[step % args.pop_size].get_arch_parameters())
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
        population.get_population()[step % args.pop_size].objs.update(loss.data, n)
        population.get_population()[step % args.pop_size].top1.update(prec1.data, n)
        population.get_population()[step % args.pop_size].top5.update(prec5.data, n)
        if (step + 1) % 100 == 0:
            logging.info("[{} Generation]".format(gen))
            logging.info(
                "Training batch #{} for {}/{}: loss={}, prec1={}, prec5={}".format(
                    step, step % args.pop_size, len(population.get_population()),
                    population.get_population()[step % args.pop_size].objs.avg,
                    population.get_population()[step % args.pop_size].top1.avg,
                    population.get_population()[step % args.pop_size].top5.avg))

# -------------------------- 验证函数（强化版） --------------------------
def validation_naswot(model, valid_queue, criterion, gen, population, device):
    model.eval()
    naswot_inputs, _ = next(iter(valid_queue))
    naswot_inputs = naswot_inputs[:args.naswot_batch_size].to(device)
    objectives = []
    for i in range(len(population.get_population())):
        individual = population.get_population()[i]
        individual.objs.reset()
        individual.top1.reset()
        individual.top5.reset()
        
        # 核心修复：使用标准化函数处理所有参数类型
        arch_params = standardize_arch_params(individual.arch_parameters, device)
        individual.arch_parameters = arch_params  # 强制更新为标准1维张量
        
        # 参数归一化（确保后续处理安全）
        arch_params = torch.clamp(arch_params, 0.0, 1.0)
        channel_candidates = torch.tensor([16, 32, 64, 128, 256], device=device)
        discrete_alphas = []
        
        # 层参数映射（严格检查索引范围）
        max_param_idx = min(28, len(arch_params))  # 确保不越界
        # 层1~8的输入通道（前8个参数）
        for j in range(min(8, max_param_idx)):
            idx = torch.argmin(torch.abs(arch_params[j] - torch.linspace(0,1,5,device=device)))
            discrete_alphas.append(channel_candidates[idx].item())
        # 层1~8的输出通道（接下来8个参数）
        for j in range(8, min(16, max_param_idx)):
            idx = torch.argmin(torch.abs(arch_params[j] - torch.linspace(0,1,5,device=device)))
            discrete_alphas.append(channel_candidates[idx].item())
        # 卷积核大小（16~20参数）
        for j in range(16, min(20, max_param_idx)):
            discrete_alphas.append(3 if arch_params[j] < 0.5 else 5)
        # 残差连接参数（20~28参数）
        for j in range(20, min(28, max_param_idx)):
            discrete_alphas.append(1 if arch_params[j] > 0.5 else 0)
        
        # 补全参数至28个（防止维度不足）
        while len(discrete_alphas) < 28:
            discrete_alphas.append(16.0)  # 缺省值
        
        # 最终标准化为1维张量
        discrete_alphas = standardize_arch_params(discrete_alphas, device)
        model.copy_arch_parameters(discrete_alphas)
        
        # 计算指标
        naswot_score = compute_naswot_score(model, naswot_inputs, device)
        genotype = model.genotype()
        params, madds = compute_params_and_flops(model, genotype, device)
        
        # 更新个体属性（确保均为标准张量）
        individual.naswot_score = naswot_score
        individual.params = params
        individual.madds = madds
        individual.arch_parameters = discrete_alphas
        
        objectives.append([-naswot_score, params, madds])
        
        # 日志输出（增加参数形状检查）
        logging.info(
            "[{} Generation] {}/{}: NASWOT={:.2f}, Params={:.2f}M, MAdds={:.2f}M, param_shape={}".format(
                gen, i + 1, len(population.get_population()),
                naswot_score, params / 1e6, madds / 1e6,
                discrete_alphas.shape))
    
    return population


# -------------------------- FGM适应度函数 --------------------------
def fgm_fitness_wrapper(model, valid_queue, device):
    def fitness_fn(arch_params):
        # 确保传入FGM的参数是标准1维张量
        standardized_params = standardize_arch_params(arch_params, device)
        model.copy_arch_parameters(standardized_params)
        naswot_inputs, _ = next(iter(valid_queue))
        naswot_inputs = naswot_inputs[:args.naswot_batch_size].to(device)
        return compute_naswot_score(model, naswot_inputs, device)
    return fitness_fn

# -------------------------- 主程序 --------------------------
def main():
    global device, population, best_accuracy, best_model_path
    # 初始化随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # 设备配置
    device = torch.device("cuda:{}".format(args.gpu))
    torch.cuda.set_device(args.gpu)
    cudnn.deterministic = True
    cudnn.enabled = True
    cudnn.benchmark = False
    # 模型和损失函数
    criterion = nn.CrossEntropyLoss().to(device)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, device)
    model.to(device)
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
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
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    # 初始化种群（强制标准化所有个体参数）
    population = Population(args.pop_size, model._steps, device)
    for individual in population.get_population():
        individual.naswot_score = -np.inf
        individual.params = float('inf')
        individual.madds = float('inf')
        # 关键：标准化新个体的架构参数
        individual.arch_parameters = standardize_arch_params(individual.arch_parameters, device)
    # 日志和目录配置
    DIR = "multi-objective-50epoch-onestep-fgm_search10-kde-{}".format(time.strftime("%Y%m%d-%H%M%S"))
    if args.dir is not None:
        if not os.path.exists(args.dir):
            utils.create_exp_dir(args.dir)
        DIR = os.path.join(args.dir, DIR)
    else:
        DIR = os.path.join(os.getcwd(), DIR)
    utils.create_exp_dir(DIR)
    utils.create_exp_dir(os.path.join(DIR, "weights"))
    # 日志配置
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(os.path.join(DIR, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger().addHandler(fh)
    logging.info("gpu device = {}".format(args.gpu))
    logging.info("args =  %s", args)
    writer = SummaryWriter(os.path.join(DIR, 'runs', 'fgm'))
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
            logging.info('[INFO] Warming up!!!')
            warm_train(model, train_queue, criterion, optimizer)
            logging.info("[INFO] Warming up finished in {:.2f} minutes".format((time.time() - start_time) / 60))
    # 初始化FGM优化器
    de_optimizer = DiffEvo(
        teacher_model=teacher_model,
        num_step=1,
        density='kde',
        noise=1.0,
        lambda_kl=args.lambda_kl if hasattr(args, 'lambda_kl') else 0.5,
        lambda_ce=args.lambda_ce if hasattr(args, 'lambda_ce') else 1.0,
        lambda_fm=args.lambda_fm,
        perturb_scale=0.1,
        kde_bandwidth=0.1
    )
    # 主训练循环
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch}: Learning rate = {scheduler.get_lr()[0]}")
        # 1. 引入新个体（强制标准化）
        if epoch % 5 == 0:
            indices = random.sample(range(len(population.get_population())), 5)
            for idx in indices:
                new_individual = population.create_new_individual()
                # 标准化新个体参数
                new_individual.arch_parameters = standardize_arch_params(new_individual.arch_parameters, device)
                population.get_population()[idx] = new_individual
            logging.info("[INFO] Introduced 5 new standardized individuals")
        # 2. 训练种群
        logging.info("[INFO] Generation {} training".format(epoch + 1))
        start_time = time.time()
        train(model, train_queue, criterion, optimizer, epoch + 1)
        logging.info("[INFO] Training finished in {:.2f} minutes".format((time.time() - start_time) / 60))
        # 3. 保存模型
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in model_state.items() if 'total_ops' not in k and 'total_params' not in k}
        torch.save(filtered_state, os.path.join(DIR, "multi-objective-cifar10-fgm-model.pt"))
        scheduler.step()
        # 4. NASWOT验证
        logging.info("[INFO] Evaluating Generation {} with NASWOT".format(epoch + 1))
        population = validation_naswot(model, valid_queue, criterion, epoch + 1, population, device)
        # 5. FGM优化（全流程标准化）
        logging.info("[INFO] Optimizing population with FGM-based DiffEvo")
        # 提取参数（已确保是1维张量）
        arch_params_list = [ind.arch_parameters.cpu().numpy() for ind in population.get_population()]
        arch_params_np = np.array(arch_params_list)
        # 准备训练数据
        train_data_batch = next(iter(train_queue))
        train_data_fgm = (train_data_batch[0].to(device), train_data_batch[1].to(device))
        # 执行FGM优化
        optimized_params, _ = de_optimizer.optimize(
            initial_population=arch_params_np,
            train_data=train_data_fgm,
            valid_queue=valid_queue,
            naswot_fn=compute_naswot_score,
            trace=True
        )
        # 更新种群参数（强制标准化）
        if isinstance(optimized_params, torch.Tensor):
            optimized_params_np = optimized_params.cpu().numpy()
        else:
            optimized_params_np = optimized_params
        
        for i, (ind, params) in enumerate(zip(population.get_population(), optimized_params_np)):
            # 最终标准化，确保万无一失
            ind.arch_parameters = standardize_arch_params(params, device)
        # 6. 更新最佳模型
        current_best = max(population.get_population(), key=lambda x: x.naswot_score)
        if current_best.naswot_score > best_accuracy:
            best_accuracy = current_best.naswot_score
            best_state = model.state_dict()
            best_filtered = {k: v for k, v in best_state.items() if 'total_ops' not in k and 'total_params' not in k}
            torch.save(best_filtered, best_model_path)
            logging.info(f"[INFO] Saved best model with NASWOT: {best_accuracy:.2f}")
            # 更新教师模型
            teacher_model.load_state_dict(best_filtered)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            de_optimizer.teacher_model = teacher_model
        # 7. 记录指标
        for i, p in enumerate(population.get_population()):
            writer.add_scalar("pop_naswot_{}".format(i + 1), p.naswot_score, epoch + 1)
            writer.add_scalar("pop_params_{}".format(i + 1), p.params, epoch + 1)
            writer.add_scalar("pop_madds_{}".format(i + 1), p.madds, epoch + 1)
        # 8. 保存种群
        with open(os.path.join(DIR, "population_{}.pickle".format(epoch + 1)), 'wb') as f:
            pickle.dump(population, f)
        # 9. 日志耗时
        epoch_time = (time.time() - start_time) / 60
        logging.info("[INFO] Epoch {}/{} finished in {:.2f} minutes".format(epoch + 1, args.epochs, epoch_time))
    # 总耗时
    total_time = (time.time() - start) / 3600
    logging.info("[INFO] Total search time: {:.2f} hours".format(total_time))
    writer.close()

if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()
    