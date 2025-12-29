from __future__ import absolute_import, division, print_function
import argparse
import archs
import datasets
from trainer.trainer_generator import GenTrainer
from trainer.trainer_utils import LinearLrDecay
from utils.utils import set_log_dir, save_checkpoint, create_logger, count_parameters_in_MB
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.flop_benchmark import print_FLOPs
from archs.super_network import Generator, Discriminator
from archs.fully_super_network import simple_Discriminator
from algorithms.diffevo_search_algs import DiffEvoSearchAlgorithm
from diffevo.fm_optimizer import DiffEvo
import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from pytorch_image_generation_metrics import get_inception_score_and_fid, get_fid, get_inception_score
import logging
from thop import profile  # 用于计算FLOPs和参数数量
import datetime

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# -------------------------- 改进的拥挤度计算方法 --------------------------
def crowding_distance(objectives, front):
    """
    改进的拥挤度计算方法，来自imagenet-doubao.py
    改进点：
    1. 用四分位距（IQR）替代max-min标准化，抗异常值；
    2. 增加局部拥挤度（k近邻），平衡全局与局部多样性；
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

# -------------------------- 改进的NSGA2-SA选择方法 --------------------------
def nsga2_sa_selection(population, objectives, pop_size, sa_current_temp=None, sa_initial_temp=None):
    """
    改进的NSGA2-SA选择方法，来自imagenet-doubao.py
    改进点：
    1. 融合拥挤度到模拟退火能量计算，优先保留"优质+多样"个体；
    2. 增加精英保留机制，强制保留前5%优质个体；
    3. 优化最后一批前沿补充逻辑，不再随机选择；
    """
    # 如果没有提供模拟退火参数，则使用默认值（兼容原有逻辑）
    if sa_current_temp is None:
        sa_current_temp = 100.0
    if sa_initial_temp is None:
        sa_initial_temp = 100.0
    
    # 1. 预处理目标函数值
    if isinstance(objectives, np.ndarray):
        fitness_list = objectives.tolist()
    else:
        fitness_list = objectives
    total_ind = len(fitness_list)
    if total_ind == 0:
        raise ValueError("待选择种群为空")

    # 2. 非支配排序
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
                # 检查i是否支配j
                dominates = True
                for k in range(len(objectives[i])):
                    if objectives[i][k] > objectives[j][k]:  # 假设所有目标都是最小化
                        dominates = False
                        break
                
                if dominates:
                    S[i].append(j)
                # 检查j是否支配i
                elif all(objectives[j][k] <= objectives[i][k] for k in range(len(objectives[j]))):
                    nP[i] += 1
            
            # 如果没有个体支配i，则i属于第一前沿
            if nP[i] == 0:
                rank[i] = 0
                fronts[0].append(i)
        
        # 构建其他前沿
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
        
        return fronts[:-1]  # 最后一个前沿是空的

    fronts = non_dominated_sorting(fitness_list)
    valid_fronts = []
    for front in fronts:
        valid_front = [int(idx) for idx in front if isinstance(idx, (int, np.integer)) and 0 <= idx < total_ind]
        if valid_front:
            valid_fronts.append(valid_front)
    if not valid_fronts:
        # 兜底：随机选择（避免原逻辑崩溃）
        return [population[i] for i in np.random.choice(total_ind, pop_size, replace=True).tolist()]

    # 3. 精英保留（强制保留前5%优质个体，确保核心性能）
    elite_ratio = 0.05
    elite_num = max(1, int(pop_size * elite_ratio))  # 至少1个精英
    elite_candidates = []
    for front in valid_fronts:
        if len(elite_candidates) >= elite_num:
            break
        # 用改进后的拥挤度计算
        crowd_dist = crowding_distance(fitness_list, front)
        # 按"非支配等级+拥挤度"排序，取前k个
        front_with_dist = list(zip(front, crowd_dist))
        front_with_dist.sort(key=lambda x: -x[1])  # 拥挤度越高越优先
        elite_candidates.extend([idx for idx, _ in front_with_dist])
    # 去重并截取精英
    elite_set = list(dict.fromkeys(elite_candidates))[:elite_num]
    selected = elite_set.copy()
    remaining = pop_size - len(selected)

    # 4. 选择剩余个体（融合拥挤度的模拟退火）
    for front in valid_fronts:
        if remaining <= 0:
            break
        # 排除已选精英
        front_non_elite = [idx for idx in front if idx not in selected]
        if not front_non_elite:
            continue

        # 若当前前沿可全部加入，直接选择
        if len(selected) + len(front_non_elite) <= pop_size:
            selected.extend(front_non_elite)
            remaining -= len(front_non_elite)
            continue

        # 5. 模拟退火选择（融合拥挤度）
        # 5.1 计算拥挤度
        crowd_dist = crowding_distance(fitness_list, front_non_elite)
        # 5.2 归一化目标值（计算能量）
        front_fitness = np.array([fitness_list[idx] for idx in front_non_elite])
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
        selected.extend(selected_from_front)

    # 6. 兜底补充（若仍不足，选拥挤度高的个体）
    if len(selected) < pop_size:
        # 收集所有未选个体
        unselected = [idx for idx in range(total_ind) if idx not in selected]
        if unselected:
            # 计算未选个体的拥挤度
            crowd_dist = crowding_distance(fitness_list, unselected)
            unselected_with_dist = list(zip(unselected, crowd_dist))
            unselected_with_dist.sort(key=lambda x: -x[1])  # 拥挤度高优先
            supplement = [idx for idx, _ in unselected_with_dist[:pop_size - len(selected)]]
            selected.extend(supplement)
        else:
            # 极端情况：随机补充
            selected.extend(np.random.choice(total_ind, pop_size - len(selected), replace=True).tolist())

    # 7. 最终校验（去重+索引有效）
    selected = list(dict.fromkeys(selected))  # 去重
    selected = [int(idx) for idx in selected if 0 <= idx < len(population)]
    # 确保种群大小
    while len(selected) < pop_size:
        selected.append(np.random.choice(selected))

    return [population[i] for i in selected[:pop_size]]

# -------------------------- NSGA-II非支配排序 --------------------------
def non_dominated_sorting(objectives):
    """
    对种群进行非支配排序
    objectives: 目标值列表，每个元素是一个包含多个目标值的元组/列表
    返回: 前沿列表，每个前沿包含个体索引
    """
    n = len(objectives)
    # 存储每个个体支配的个体
    S = [[] for _ in range(n)]
    # 支配每个个体的个体数量
    nP = [0] * n
    # 个体所在的前沿
    rank = [0] * n
    # 前沿列表
    fronts = [[]]
    
    # 计算支配关系
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # 检查i是否支配j
            dominates = True
            for k in range(len(objectives[i])):
                if objectives[i][k] > objectives[j][k]:  # 假设所有目标都是最小化
                    dominates = False
                    break
            
            if dominates:
                S[i].append(j)
            # 检查j是否支配i
            elif all(objectives[j][k] <= objectives[i][k] for k in range(len(objectives[j]))):
                nP[i] += 1
        
        # 如果没有个体支配i，则i属于第一前沿
        if nP[i] == 0:
            rank[i] = 0
            fronts[0].append(i)
    
    # 构建其他前沿
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
    
    return fronts[:-1]  # 最后一个前沿是空的

# -------------------------- 计算参数数量和FLOPs --------------------------
def compute_params_and_flops(model, input_shape=(1, 120), device='cuda', genotype=None):
    """计算生成器模型的参数数量和FLOPs"""
    model.eval()
    with torch.no_grad():
        # 创建适当形状的随机输入
        dummy_input = torch.randn(input_shape).to(device)

        # 将genotype转换为张量（如果需要）
        if genotype is not None and isinstance(genotype, np.ndarray):
            genotype_tensor = torch.from_numpy(genotype).long().to(device)
        else:
            genotype_tensor = genotype

        # 计算MACs和参数数量
        macs, params = profile(model, inputs=(dummy_input, genotype_tensor), verbose=False)
        madds = macs * 2  # 假设MACs乘以2得到MAdds

        return params, madds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_selected', type=int, default=10, help='number of selected genotypes')
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset type')
    parser.add_argument('--loss', type=str, default='hinge', help='loss function')
    parser.add_argument('--img_size', type=int, default=32, help='image size, 32 for cifar10, 48 for stl10')
    parser.add_argument('--bottom_width', type=int, default=4, help='init resolution, 4 for cifar10, 6 for stl10')
    parser.add_argument('--channels', type=int, default=3, help='image channels')
    parser.add_argument('--data_path', type=str, default='data/datasets/cifar-10', help='dataset path')

    parser.add_argument('--exp_name', type=str, default='arch_searchG_cifar10-fm', help='experiment name')
    parser.add_argument('--gpu_ids', type=str, default="0", help='visible GPU ids')
    parser.add_argument('--num_workers', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path')
    
    # train
    parser.add_argument('--arch', type=str, default='arch_cifar10', help='architecture name')
    parser.add_argument('--my_max_epoch_G', type=int, default=40, help='max number of epoch for training G')
    parser.add_argument('--max_iter_G', type=int, default=None, help='max number of iteration for training G')
    parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
    parser.add_argument('--gen_bs', type=int, default=256, help='batche size of G')
    parser.add_argument('--dis_bs', type=int, default=128, help='batche size of D')
    parser.add_argument('--gf_dim', type=int, default=256, help='base channel-dim of G')
    parser.add_argument('--df_dim', type=int, default=128, help='base channel-dim of D')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--lr_decay', action='store_true', help='learning rate decay or not')
    parser.add_argument('--beta1', type=float, default=0.0, help='decay of first order momentum of gradient')
    parser.add_argument('--beta2', type=float, default=0.9, help='decay of first order momentum of gradient')
    parser.add_argument('--init_type', type=str, default='xavier_uniform',
                        choices=['normal', 'orth', 'xavier_uniform', 'false'],
                        help='init type')
    parser.add_argument('--d_spectral_norm', type=str2bool, default=True,
                        help='add spectral_norm on discriminator or not')
    parser.add_argument('--g_spectral_norm', type=str2bool, default=False,
                        help='add spectral_norm on generator or not')
    parser.add_argument('--latent_dim', type=int, default=120, help='dimensionality of the latent space')

    # val
    parser.add_argument('--print_freq', type=int, default=100, help='interval between each verbose')
    parser.add_argument('--val_freq', type=int, default=5, help='interval between each validation')
    parser.add_argument('--num_eval_imgs', type=int, default=100)
    parser.add_argument('--eval_batch_size', type=int, default=100)
    parser.add_argument('--mute_max_num', type=int, default=1, help='max number of mutations per individual')
    # search
    parser.add_argument('--derived_start_epoch', type=int, default=0, help='')
    parser.add_argument('--derived_max_epoch', type=int, default=None, help='')
    parser.add_argument('--derived_epoch_interval', type=int, default=None, help='')
    parser.add_argument('--tau_max', type=float, default=5, help='max tau for gumbel softmax')
    parser.add_argument('--tau_min', type=float, default=0.1, help='min tau for gumbel softmax')
    parser.add_argument('--gumbel_softmax', type=str2bool, default=False, help='use gumbel softmax or not')
    parser.add_argument('--amending_coefficient', type=float, default=0.1, help='')
    parser.add_argument('--derive_freq', type=int, default=1, help='')
    parser.add_argument('--derive_per_epoch', type=int, default=0, help='number of derive per epoch')
    parser.add_argument('--draw_arch', type=str2bool, default=False, help='visualize the searched architecture or not')
    parser.add_argument('--early_stop', type=str2bool, default=False, help='use early stop strategy or not')
    parser.add_argument('--genotypes_exp', type=str, default='default_genotype.npy', help='ues genotypes of the experiment')
    parser.add_argument('--cr', type=float, default=0.0)
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema')
    parser.add_argument('--genotype_of_G', type=str, default='best_G.npy', help='ues genotypes of the experiment')
    parser.add_argument('--use_basemodel_D', type=str2bool, default=False, help='use the base model of D')

    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--base_arch', type=str, default='None', help='base arch for train')

    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--Total_evolutionary_algebra', type=int, default=20)
    parser.add_argument('--num_individual', type=int, default=20)
    parser.add_argument('--num_op_g', type=int, default=1)
    parser.add_argument('--max_model_size', type=int, default=13)

    # 多目标搜索参数
    parser.add_argument('--multi_objective', type=str2bool, default=True, help='使用多目标优化')
    parser.add_argument('--obj_is_weight', type=float, default=1.0, help='IS目标权重（最大化IS，因此取负）')
    parser.add_argument('--obj_fid_weight', type=float, default=1.0, help='FID目标权重')
    parser.add_argument('--obj_param_weight', type=float, default=0.1, help='参数数量目标权重')
    parser.add_argument('--obj_madds_weight', type=float, default=0.01, help='MAdds目标权重')

    opt = parser.parse_args()

    return opt

def main():
    args = parse_args()
    torch.cuda.manual_seed(args.random_seed)
    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for id in range(len(str_ids)):
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 1:
        args.gpu_ids = args.gpu_ids[1:]
    else:
        args.gpu_ids = args.gpu_ids

    # 先设置路径 helper，确保日志目录存在
    args.path_helper = set_log_dir('exps', args.exp_name)
    
    # 创建日志记录器，将日志保存到 exps 目录下的实验日志路径
    logger = create_logger(args.path_helper['log_path'])
    logger.info("开始多目标架构搜索")
    logger.info("参数: %s", args)

    # genotype G
    search_alg = DiffEvoSearchAlgorithm(args)

    # import network from genotype
    basemodel_gen = Generator(args)
    gen_net = torch.nn.DataParallel(
        basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])
    basemodel_dis = simple_Discriminator()
    dis_net = torch.nn.DataParallel(
        basemodel_dis, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, 1.)
            else:
                raise NotImplementedError(
                    '{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net.apply(weights_init)
    dis_net.apply(weights_init)

    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train

    max_epoch_for_D = args.my_max_epoch_G * args.n_critic
    args.max_epoch_D = args.my_max_epoch_G * args.n_critic
    max_iter_D = max_epoch_for_D * len(train_loader)

    if args.dataset.lower() == 'cifar10':
        fid_stat = './fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = './fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)

    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, max_iter_D)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, max_iter_D)

    start_epoch = 0
    best_fid = 1e4

    # 初始化writer_dict，使用exps目录下的日志路径
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    logger.info('Param size of G = %fMB', count_parameters_in_MB(gen_net))
    logger.info('Param size of D = %fMB', count_parameters_in_MB(dis_net))

    genotype_fixG = np.load(os.path.join('exps', 'best_G.npy'))
    trainer_gen = GenTrainer(args, gen_net, dis_net, gen_optimizer,
                             dis_optimizer, train_loader, search_alg, None,
                             genotype_fixG)

    best_genotypes = None
    is_mean_best = 0.0
    fid_mean_best = 999.0

    temp_model_size = args.max_model_size
    args.max_model_size = 999

    epoch_record = []
    is_record = []
    fid_record = []
    params_record = []
    madds_record = []

    # 多目标评估函数
    def multi_objective_evaluation(genotypes, fid_stat, gen_net, args, generation_idx=None):
        """多目标评估函数，返回FID、参数数量和MAdds"""
        objectives = []
        raw_metrics = []  # 存储原始指标 (is_score, fid_score, params, madds)

        for idx, genotype in enumerate(genotypes):
            # 评估FID - 使用validate方法而不是evaluate_genotype
            is_score, _, fid_score = trainer_gen.validate(genotype, fid_stat)

            # 计算参数数量和MAdds
            # 确保基因型是3x7的numpy数组
            if isinstance(genotype, np.ndarray) and genotype.shape == (3, 7):
                try:
                    # 使用正确的参数创建Generator
                    gen_net_from_genotype = Generator(args)

                    # 设置基因型（如果Generator类有相应的方法）
                    if hasattr(gen_net_from_genotype, 'set_genotype'):
                        gen_net_from_genotype.set_genotype(genotype)
                    elif hasattr(gen_net_from_genotype, 'genotype'):
                        gen_net_from_genotype.genotype = genotype
                    else:
                        # 如果无法设置基因型，尝试使用反射来设置细胞基因型
                        try:
                            for i in range(3):
                                cell_attr = f'cell{i + 1}'
                                if hasattr(gen_net_from_genotype, cell_attr):
                                    cell = getattr(gen_net_from_genotype, cell_attr)
                                    if hasattr(cell, 'set_genotype'):
                                        cell.set_genotype(genotype[i])
                                    elif hasattr(cell, 'genotype'):
                                        cell.genotype = genotype[i]
                        except Exception as e:
                            logger.error(f"设置细胞基因型失败: {e}")
                            objectives.append([float('inf'), float('inf'), float('inf'), float('inf')])
                            raw_metrics.append((0, float('inf'), float('inf'), float('inf')))
                            continue

                    gen_net_from_genotype = gen_net_from_genotype.cuda(args.gpu_ids[0])

                    params, madds = compute_params_and_flops(
                        gen_net_from_genotype,
                        input_shape=(1, args.latent_dim),
                        device=f'cuda:{args.gpu_ids[0]}',
                        genotype=genotype
                    )

                    # 四目标：最大化IS（因此取负），最小化FID、参数数量和MAdds
                    objective_value = [
                        -is_score * args.obj_is_weight,  # 最大化IS等价于最小化-IS
                        fid_score * args.obj_fid_weight,
                        params * args.obj_param_weight,
                        madds * args.obj_madds_weight
                    ]

                    objectives.append(objective_value)
                    raw_metrics.append((is_score, fid_score, params, madds))

                    # 记录每个个体的详细信息
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logger.info(f"{current_time} 基因型{idx}评估 - IS: {is_score:.3f}, FID: {fid_score:.3f}, "
                               f"Params: {params / 1e6:.3f}M, MAdds: {madds / 1e6:.3f}M")

                except Exception as e:
                    logger.error(f"创建Generator失败: {e}")
                    objectives.append([float('inf'), float('inf'), float('inf'), float('inf')])
                    raw_metrics.append((0, float('inf'), float('inf'), float('inf')))
            else:
                logger.warning(
                    f"基因型格式不正确: {type(genotype)}, 形状: {genotype.shape if hasattr(genotype, 'shape') else 'N/A'}")
                objectives.append([float('inf'), float('inf'), float('inf'), float('inf')])
                raw_metrics.append((0, float('inf'), float('inf'), float('inf')))

        # 记录整个种群的统计信息
        if raw_metrics and generation_idx is not None:
            is_scores = [m[0] for m in raw_metrics]
            fid_scores = [m[1] for m in raw_metrics]
            params_scores = [m[2] for m in raw_metrics]
            madds_scores = [m[3] for m in raw_metrics]
            
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"{current_time} mean_IS: {np.mean(is_scores):.3f}, mean_FID: {np.mean(fid_scores):.3f}, "
                       f"max_IS: {np.max(is_scores):.3f}, min_FID: {np.min(fid_scores):.3f}, "
                       f"mean_params: {np.mean(params_scores) / 1e6:.3f}M, "
                       f"mean_MAdds: {np.mean(madds_scores) / 1e6:.3f}M @ evolutionary_algebra: {generation_idx}.")

        return objectives, raw_metrics

    for epoch in tqdm(range(int(start_epoch), int(200)), desc='training the supernet_G:'):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        trainer_gen.train(epoch, writer_dict, fid_stat, lr_schedulers)
        if epoch == 200:
            break
        if epoch >= 9999 and epoch % 5 == 0:
            now_is_max = 0
            now_fid_min = 999
            now_params_min = float('inf')
            now_madds_min = float('inf')
            
            trainer_gen.clear_bag()
            # 生成初始种群：每个基因型展平为21维，堆叠为 (12, 21)
            population = np.stack(
                [search_alg.sample_fair().flatten() for i in range(12)],
                axis=0
            )  # 形状：(12, 21)

            optimizer = DiffEvo(
                teacher_model=gen_net,
                num_step=1,
                density='kde',
                noise=1.0,
                lambda_kl=args.lambda_kl if hasattr(args, 'lambda_kl') else 0.5,
                lambda_ce=args.lambda_ce if hasattr(args, 'lambda_ce') else 1.0,
                lambda_fm=1.0,
                perturb_scale=0.1,
                kde_bandwidth=0.1
            )

            train_data = next(iter(train_loader))
            # 优化器应返回 (12, 21) 的二维张量
            result = optimizer.optimize(population, train_data, gen_net, fid_stat)
            # 检查返回值是否为元组，如果是则提取第一个元素
            if isinstance(result, tuple):
                population = result[0]
            else:
                population = result

            # 确保种群维度正确：(n_samples, 21)
            assert population.ndim == 2, f"Population must be 2D, got {population.ndim}D"
            assert population.shape[1] == 21, f"Expected 21 features per genotype, got {population.shape[1]}"

            for kk in tqdm(range(4), desc='Evaluating of subnet performance using evolutionary algorithms'):
                # 将每个21维向量恢复为 (3, 7)
                reshaped_population = [g.reshape(3, 7) for g in population]
                
                if args.multi_objective:
                    # 多目标评估
                    objectives, raw_metrics = multi_objective_evaluation(reshaped_population, fid_stat, gen_net, args, kk)
                    
                    # 记录最佳个体信息
                    if raw_metrics:
                        best_idx = min(range(len(objectives)), key=lambda i: sum(objectives[i]))
                        best_is, best_fid, best_params, best_madds = raw_metrics[best_idx]
                        
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        logger.info(f"{current_time} 最佳个体 - IS: {best_is:.3f}, FID: {best_fid:.3f}, "
                                   f"Params: {best_params / 1e6:.3f}M, MAdds: {best_madds / 1e6:.3f}M "
                                   f"@ evolutionary_algebra: {kk}.")
                    
                    # 使用改进的NSGA2-SA进行多目标选择
                    selected_indices = nsga2_sa_selection(
                        list(range(len(reshaped_population))),
                        objectives,
                        len(reshaped_population) // 2  # 选择一半个体
                    )
                    
                    # 更新种群
                    population = np.stack([population[i] for i in selected_indices], axis=0)
                    
                    # 记录最佳值
                    if raw_metrics:
                        best_idx = min(range(len(objectives)), key=lambda i: sum(objectives[i]))
                        now_is_max = max(now_is_max, raw_metrics[best_idx][0])
                        now_fid_min = min(now_fid_min, raw_metrics[best_idx][1])
                        now_params_min = min(now_params_min, raw_metrics[best_idx][2])
                        now_madds_min = min(now_madds_min, raw_metrics[best_idx][3])
                else:
                    # 单目标评估（原有逻辑）
                    population, pop_selected, is_mean, fid_mean, is_max, fid_min = trainer_gen.my_search_evol(
                        reshaped_population, fid_stat, kk
                    )
                    # 评估后重新展平为 (n_samples, 21)
                    population = np.stack([g.flatten() for g in population], axis=0)
                    if is_max > now_is_max:
                        now_is_max = is_max
                    if fid_min < now_fid_min:
                        now_fid_min = fid_min

            epoch_record.append(epoch)
            is_record.append(now_is_max)
            fid_record.append(now_fid_min)
            params_record.append(now_params_min)
            madds_record.append(now_madds_min)
            
            np.save('epoch_record_518.npy', np.array(epoch_record))
            np.save('is_record_518.npy', np.array(is_record))
            np.save('fid_record_518.npy', np.array(fid_record))
            np.save('params_record_518.npy', np.array(params_record))
            np.save('madds_record_518.npy', np.array(madds_record))

            trainer_gen.clear_bag()

            if is_mean > is_mean_best:
                is_mean_best = is_mean
                checkpoint_file = os.path.join(args.path_helper['ckpt_path'], 'gen_checkpoint_best_is.pt')
                ckpt = {'epoch': epoch,
                        'weight_G': gen_net.state_dict(),
                        'weight_D': dis_net.state_dict(),
                        'up_G_fixed': search_alg.Up_G_fixed,
                        'normal_G_fixed': search_alg.Normal_G_fixed,
                        }
                torch.save(ckpt, checkpoint_file)
                del ckpt
            if fid_mean < fid_mean_best:
                fid_mean_best = fid_mean
                checkpoint_file = os.path.join(args.path_helper['ckpt_path'], 'gen_checkpoint_best_fid.pt')
                ckpt = {'epoch': epoch,
                        'weight_G': gen_net.state_dict(),
                        'weight_D': dis_net.state_dict(),
                        'up_G_fixed': search_alg.Up_G_fixed,
                        'normal_G_fixed': search_alg.Normal_G_fixed,
                        }
                torch.save(ckpt, checkpoint_file)
                del ckpt

        if epoch == args.warmup * args.n_critic:
            checkpoint_file = os.path.join(args.path_helper['ckpt_path'], 'gen_checkpoint_before_prune.pt')
            ckpt = {'epoch': epoch,
                    'weight_G': gen_net.state_dict(),
                    'weight_D': dis_net.state_dict(),
                    'up_G_fixed': search_alg.Up_G_fixed,
                    'normal_G_fixed': search_alg.Normal_G_fixed,
                    }
            torch.save(ckpt, checkpoint_file)
            del ckpt
            trainer_gen.directly_modify_fixed(fid_stat)
            logger.info(
                f'search_alg.Normal_G_fixed: {search_alg.Normal_G_fixed}, search_alg.Up_G_fixed: {search_alg.Up_G_fixed},@ epoch {epoch}.')

    checkpoint_file = os.path.join(args.path_helper['ckpt_path'], 'supernet_gen.pt')
    ckpt = {
        'weight_G': gen_net.state_dict(),
        'up_G_fixed': search_alg.Up_G_fixed,
        'normal_G_fixed': search_alg.Normal_G_fixed,
    }
    torch.save(ckpt, checkpoint_file)
    args.max_model_size = temp_model_size

    # 最终搜索阶段
    population = np.stack(
        [search_alg.sample_fair().flatten() for i in range(args.num_individual)],
        axis=0
    )  # 形状：(num_individual, 21)

    optimizer = DiffEvo(
        teacher_model=gen_net,
        num_step=1,
        density='kde',
        noise=1.0,
        lambda_kl=args.lambda_kl if hasattr(args, 'lambda_kl') else 0.5,
        lambda_ce=args.lambda_ce if hasattr(args, 'lambda_ce') else 1.0,
        lambda_fm=1.0,
        perturb_scale=0.1,
        kde_bandwidth=0.1
    )
    train_data = next(iter(train_loader))
    
    # 存储多目标评估结果
    all_objectives = []
    all_genotypes = []
    all_raw_metrics = []
    
    for ii in tqdm(range(args.Total_evolutionary_algebra), desc='search genearator using evo alg'):
        result = optimizer.optimize(population, train_data, gen_net, fid_stat)
        if isinstance(result, tuple):
            population = result[0]
        else:
            population = result
        # 验证维度
        assert population.ndim == 2 and population.shape[1] == 21, "Invalid genotype dimension"

        # 转换为 numpy 数组并 reshape
        reshaped_population = []
        for g in population:
            # 处理可能的 torch 张量（先移到CPU再转换为 numpy 数组）
            if isinstance(g, torch.Tensor):
                g_np = g.cpu().numpy()  # 先转移到CPU再转numpy
            else:
                g_np = g
            reshaped = g_np.reshape(3, 7)
            assert reshaped.shape == (3, 7), f"Genotype shape error: {reshaped.shape}, expected (3, 7)"
            reshaped_population.append(reshaped)
         
        if args.multi_objective:
            # 多目标评估
            objectives, raw_metrics = multi_objective_evaluation(reshaped_population, fid_stat, gen_net, args, ii)
            all_objectives.extend(objectives)
            all_genotypes.extend(reshaped_population)
            all_raw_metrics.extend(raw_metrics)
            
            # 使用改进的NSGA2-SA进行选择
            selected_indices = nsga2_sa_selection(
                list(range(len(reshaped_population))),
                objectives,
                len(reshaped_population)  # 保持种群大小不变
            )
            
            # 更新种群
            population = np.stack([population[i].cpu().numpy() for i in selected_indices], axis=0)
        else:
            # 单目标评估（原有逻辑）
            population, pop_selected, is_mean, fid_mean, _, _ = trainer_gen.my_search_evolv2(
                reshaped_population, fid_stat, ii
            )
            # 重新展平为 21 维向量
            population = np.stack([g.flatten() for g in population], axis=0)
    
    # 多目标优化后，选择Pareto前沿上的解
    if args.multi_objective and all_objectives:
        # 找到Pareto前沿
        fronts = non_dominated_sorting(all_objectives)
        pareto_front = fronts[0] if fronts else []
        
        # 选择Pareto前沿上的个体
        pop_selected = [all_genotypes[i] for i in pareto_front]
        
        logger.info(f"找到 {len(pareto_front)} 个Pareto最优解")
        for i, idx in enumerate(pareto_front):
            is_score, fid_score, params, madds = all_raw_metrics[idx]
            logger.info(f"解 {i+1}: IS={is_score:.4f}, FID={fid_score:.4f}, "
                       f"Params={params / 1e6:.2f}M, MAdds={madds / 1e6:.2f}M")
    else:
        # 单目标选择
        pop_selected = pop_selected if 'pop_selected' in locals() else []

    for index, geno in enumerate(pop_selected):
        file_path = os.path.join(args.path_helper['ckpt_path'],
                                 "best_gen_{}.npy".format(str(index)))
        np.save(file_path, geno)

if __name__ == '__main__':
    main()