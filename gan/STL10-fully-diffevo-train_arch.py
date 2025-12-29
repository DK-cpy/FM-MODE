from __future__ import absolute_import, division, print_function
import argparse
import archs
import datasets
from network import train, my_validate, LinearLrDecay, load_params, copy_params, save_dis_images
from utils.utils import set_log_dir, save_checkpoint, save_is_checkpoint, create_logger, count_parameters_in_MB, \
    save_epoch_checkpoint
# from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.flop_benchmark import print_FLOPs
from archs.fully_super_network import Generator, Discriminator
import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from archs.fully_super_network import simple_Discriminator
from archs.network_64 import Discriminator64

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--dataset', type=str, default='stl10', help='dataset type')
    parser.add_argument('--loss', type=str, default='hinge', help='loss function')
    parser.add_argument('--img_size', type=int, default=48, help='image size, 32 for cifar10, 48 for stl10')
    parser.add_argument('--bottom_width', type=int, default=6, help='init resolution, 4 for cifar10, 6 for stl10')
    parser.add_argument('--channels', type=int, default=3, help='image channels')
    parser.add_argument('--data_path', type=str, default='/data/datasets/stl10', help='dataset path')

    parser.add_argument('--exp_name', type=str, default='arch_train_stl10', help='experiment name')
    parser.add_argument('--gpu_ids', type=str, default="0", help='visible GPU ids')
    parser.add_argument('--num_workers', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path')

    # train
    parser.add_argument('--arch', type=str, default='arch_stl10', help='architecture name')
    # parser.add_argument('--arch_D', type=str, help='architecture name of D')
    parser.add_argument('--max_epoch_G', type=int, default=600, help='max number of epoch for training G')
    parser.add_argument('--max_iter_G', type=int, default=None, help='max number of iteration for training G')
    parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
    parser.add_argument('--gen_bs', type=int, default=256, help='batche size of G')
    parser.add_argument('--dis_bs', type=int, default=128, help='batche size of D')
    parser.add_argument('--gf_dim', type=int, default=256, help='base channel-dim of G')
    parser.add_argument('--df_dim', type=int, default=128, help='base channel-dim of D')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for D')
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
    parser.add_argument('--num_eval_imgs', type=int, default=50000)
    parser.add_argument('--eval_batch_size', type=int, default=100)

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
    parser.add_argument('--genotypes_exp', type=str, default='stl10_D.npy', help='ues genotypes of the experiment')
    parser.add_argument('--cr', type=float, default=1)
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema')
    parser.add_argument('--genotype_of_G', type=str, default='arch_searchG_cifar10_2025_06_03_14_22_27/Model/best_fid_gen.npy', help='ues genotypes of the experiment')
    parser.add_argument('--use_basemodel_D', type=str2bool, default=False, help='use the base model of D')

    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--base_arch', type=str, default='None', help='base arch for train')

    opt = parser.parse_args()

    return opt

def main():
    args = parse_args()
    torch.cuda.manual_seed(args.random_seed)

    # set visible GPU ids
    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # set TensorFlow environment for evaluation (calculate IS and FID)
    # _init_inception()
    # inception_path = check_or_download_inception('./tmp/imagenet/')
    # create_inception_graph(inception_path)

    # the first GPU in visible GPUs is dedicated for evaluation (running Inception model)
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for id in range(len(str_ids)):
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 1:
        args.gpu_ids = args.gpu_ids[1:]
    else:
        args.gpu_ids = args.gpu_ids
    genotype_G = np.load(os.path.join('exps', args.genotype_of_G))

    # 载入模型
    basemodel_gen = Generator(args, genotype_G)
    gen_net = torch.nn.DataParallel(basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    if args.use_basemodel_D:
        basemodel_dis = Discriminator64(args)
        # basemodel_dis = simple_Discriminator()

    else:
        print("Loading genotypes from:", args.genotypes_exp)
        genotype_D = np.load(os.path.join('exps', args.genotypes_exp))
        basemodel_dis = Discriminator(args, genotype_D)

    dis_net = torch.nn.DataParallel(basemodel_dis, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    # weight init
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
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net.apply(weights_init)
    dis_net.apply(weights_init)

    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train

    # epoch number for dis_net
    args.max_epoch_D = args.max_epoch_G * args.n_critic
    if args.max_iter_G:
        args.max_epoch_D = np.ceil(args.max_iter_G * args.n_critic / len(train_loader))
    max_iter_D = args.max_epoch_D * len(train_loader)

    # set optimizer
    if args.optimizer.lower() == 'adam':
        gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                         args.g_lr, (args.beta1, args.beta2))
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                         args.d_lr, (args.beta1, args.beta2))
    elif args.optimizer.lower() == 'rmsprop':
        gen_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, gen_net.parameters()), args.g_lr,
                                            alpha=0.9)
        dis_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, dis_net.parameters()), args.d_lr,
                                            alpha=0.9)
    elif args.optimizer.lower() == 'sgd':
        gen_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, gen_net.parameters()), lr=args.g_lr,
                                        momentum=0.8)
        dis_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, dis_net.parameters()), lr=args.d_lr,
                                        momentum=0.8)
    else:
        raise NotImplementedError(
            'Unknown optimizer: {}'.format(args.optimizer))
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, max_iter_D)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, max_iter_D)

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = './fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = './fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    elif args.dataset.lower() == 'celeba':
        fid_stat = './fid_stat/celebahq.3k.128.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)

    # initial
    gen_avg_param = copy_params(gen_net)
    start_epoch = 0
    best_fid = 1e4
    best_is = 0.
    # set writer
    if args.checkpoint:
        # resuming
        print(f'=> resuming from {args.checkpoint}')
        assert os.path.exists(os.path.join('exps', args.checkpoint))
        checkpoint_file = os.path.join('exps', args.checkpoint, 'Model', 'checkpoint_best.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint['best_fid']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        avg_gen_net = deepcopy(gen_net)
        avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(avg_gen_net)
        del avg_gen_net

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('exps', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    # model size
    logger.info('Param size of G = %fMB', count_parameters_in_MB(gen_net))
    logger.info('Param size of D = %fMB', count_parameters_in_MB(dis_net))
    print_FLOPs(basemodel_gen, (1, args.latent_dim), logger)
    print_FLOPs(basemodel_dis, (1, 3, args.img_size, args.img_size), logger)

    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))

    # train loop
    for epoch in tqdm(range(int(start_epoch), int(args.max_epoch_D)), desc='total progress'):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer,
              gen_avg_param, train_loader, epoch, writer_dict, lr_schedulers)
        epoch_g = epoch // args.n_critic
        if epoch == 2111:
            break
        #if (epoch % args.val_freq == 0 and epoch > 500) or epoch == 2200:
        if (epoch % args.val_freq == 0 and epoch > 400) or epoch == 210 or epoch ==100 or epoch==300:

            inception_score, std, fid_score = my_validate(args, gen_net, fid_stat)
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param)
            inception_score_ema, std_ema, fid_score_ema = my_validate(args, gen_net, fid_stat)
            logger.info(f'IS: {round(inception_score, 3)}, IS_std:{round(std, 3)}, FID:{round(fid_score, 3)}, '
                        f'IS_ema: {round(inception_score_ema, 3)}, IS_std_ema: {round(std_ema, 3)}, '
                        f'FID_ema: {round(fid_score_ema, 3)} || @ epoch {epoch}.')
            load_params(gen_net, backup_param)
            # early stop
            if fid_score_ema >= 30:
                break
            if fid_score_ema < best_fid:
                best_fid = fid_score_ema
                is_best = True
            else:
                is_best = False
            if inception_score_ema > best_is:
                best_is = inception_score_ema
                is_best_is = True
            else:
                is_best_is = False
        else:
            is_best = False
            is_best_is = False
        # save model
        #save_condition = epoch_g % args.val_freq == 0 and epoch_g >= 0
        if epoch_g % args.val_freq == 0 and epoch >= 400:
            avg_gen_net = deepcopy(gen_net)
            load_params(avg_gen_net, gen_avg_param)
            save_is_checkpoint({
                'epoch': epoch + 1,
                'model': args.arch,
                'gen_state_dict': gen_net.state_dict(),
                'dis_state_dict': dis_net.state_dict(),
                'avg_gen_state_dict': avg_gen_net.state_dict(),
                'gen_optimizer': gen_optimizer.state_dict(),
                'dis_optimizer': dis_optimizer.state_dict(),
                'best_fid': best_fid,
                'path_helper': args.path_helper
            }, is_best, args.path_helper['ckpt_path'])
            del avg_gen_net
        if epoch_g % 2 == 0 and epoch >= 300:
            avg_gen_net = deepcopy(gen_net)
            load_params(avg_gen_net, gen_avg_param)
            save_epoch_checkpoint({
                'epoch': epoch + 1,
                'model': args.arch,
                'gen_state_dict': gen_net.state_dict(),
                'dis_state_dict': dis_net.state_dict(),
                'avg_gen_state_dict': avg_gen_net.state_dict(),
                'gen_optimizer': gen_optimizer.state_dict(),
                'dis_optimizer': dis_optimizer.state_dict(),
                'path_helper': args.path_helper
            }, epoch, args.path_helper['ckpt_path'])
            del avg_gen_net
        if 2984< epoch < 3005 or 600< epoch < 699 or 300< epoch < 399:  #% 9999 == 0  #<9999
            save_dis_images(args, gen_net, epoch, False)

            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param)
            save_dis_images(args, gen_net, epoch, True)
            load_params(gen_net, backup_param)


if __name__ == '__main__':
    main()