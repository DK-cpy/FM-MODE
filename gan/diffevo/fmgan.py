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

# -------------------------- NSGA-II非支配排序和拥挤度计算 --------------------------
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

def crowding_distance(objectives, front):
    """
    计算前沿中个体的拥挤度
    objectives: 所有个体的目标值
    front: 当前前沿的个体索引列表
    返回: 拥挤度值列表
    """
    n = len(front)
    if n == 0:
        return []
    
    # 初始化拥挤度
    distances = [0.0] * n
    
    # 对每个目标计算拥挤度
    num_objectives = len(objectives[0])
    for m in range(num_objectives):
        # 按当前目标值排序
        sorted_front = sorted(front, key=lambda i: objectives[i][m])
        
        # 设置边界个体的拥挤度为无穷大
        distances[0] = float('inf')
        distances[-1] = float('inf')
        
        # 获取当前目标的最大最小值
        min_val = objectives[sorted_front[0]][m]
        max_val = objectives[sorted_front[-1]][m]
        
        if max_val - min_val < 1e-6:
            continue
            
        # 计算中间个体的拥挤度
        for i in range(1, n-1):
            idx = sorted_front[i]
            next_val = objectives[sorted_front[i+1]][m]
            prev_val = objectives[sorted_front[i-1]][m]
            distances[i] += (next_val - prev_val) / (max_val - min_val)
    
    return distances

def nsga2_selection(population, objectives, pop_size):
    """
    NSGA2选择过程
    population: 种群列表
    objectives: 每个个体的目标值列表
    pop_size: 需要选择的个体数量
    返回: 被选中的个体列表
    """
    # 非支配排序
    fronts = non_dominated_sorting(objectives)
    
    # 选择个体直到达到种群大小
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            # 整个前沿都被选择
            selected.extend(front)
        else:
            # 需要部分选择这个前沿
            # 计算拥挤度
            distances = crowding_distance(objectives, front)
            
            # 按拥挤度排序（降序）
            sorted_front = sorted(zip(front, distances), key=lambda x: -x[1])
            
            # 选择拥挤度最高的个体
            remaining = pop_size - len(selected)
            selected.extend([idx for idx, _ in sorted_front[:remaining]])
            break
    
    return [population[i] for i in selected]

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

    parser.add_argument('--exp_name', type=str, default='arch_searchG_cifar10', help='experiment name')
    parser.add_argument('--gpu_ids', type=str, default="0", help='visible GPU ids')
    parser.add_argument('--num_workers', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path')
    
    # train
    parser.add_argument('--arch', type=str, default='arch_cifar10', help='architecture name')
    parser.add_argument('--my_max_epoch_G', type=int, default=1, help='max number of epoch for training G')
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
    parser.add_argument('--lambda_fm', type=float, default=1.0, help='权重参数 for FGM损失')

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
    parser.add_argument('--Total_evolutionary_algebra', type=int, default=2)
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
    logger.info("开始多目标GAN架构搜索")
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

    for epoch in tqdm(range(int(start_epoch), int(2)), desc='training the supernet_G:'):
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
                lambda_fm=args.lambda_fm,
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
                    
                    # 使用NSGA-II进行多目标选择
                    selected_indices = nsga2_selection(
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
        lambda_fm=args.lambda_fm,
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
            # 处理可能的 torch 张量（转换为 numpy 数组）
            g_np = g.numpy() if isinstance(g, torch.Tensor) else g
            reshaped = g_np.reshape(3, 7)
            assert reshaped.shape == (3, 7), f"Genotype shape error: {reshaped.shape}, expected (3, 7)"
            reshaped_population.append(reshaped)
         
        if args.multi_objective:
            # 多目标评估
            objectives, raw_metrics = multi_objective_evaluation(reshaped_population, fid_stat, gen_net, args, ii)
            all_objectives.extend(objectives)
            all_genotypes.extend(reshaped_population)
            all_raw_metrics.extend(raw_metrics)
            
            # 使用NSGA-II进行选择
            selected_indices = nsga2_selection(
                list(range(len(reshaped_population))),
                objectives,
                len(reshaped_population)  # 保持种群大小不变
            )
            
            # 更新种群
            population = np.stack([population[i] for i in selected_indices], axis=0)
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