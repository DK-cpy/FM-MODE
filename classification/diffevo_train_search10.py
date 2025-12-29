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
from diffevo import DiffEvo  # 假设 DiffEvo 类位于 diffusion_evo.py 文件中

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/fasterdatasets/cifar-10', help='location of the data corpus')
parser.add_argument('--dir', type=str, default=None, help='location of trials')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=3, help='num of training epochs')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--gpu', type=int, default=2, help='gpu device id')
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
args = parser.parse_args()

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
                "Using Training batch #{} for {}/{} architecture with loss: {}, prec1: {}, prec5: {}".format(step,
                                                                                                             step % args.pop_size,
                                                                                                             len(population.get_population()),
                                                                                                             population.get_population()[
                                                                                                                 step % args.pop_size].objs.avg,
                                                                                                             population.get_population()[
                                                                                                                 step % args.pop_size].top1.avg,
                                                                                                             population.get_population()[
                                                                                                                 step % args.pop_size].top5.avg))

def validation(model, valid_queue, criterion, gen, population):
    model.eval()
    for i in range(len(population.get_population())):
        individual = population.get_population()[i]
        #print(f"Validation: Individual {i} arch_parameters shape {individual.arch_parameters.shape}")
        individual.objs.reset()
        individual.top1.reset()
        individual.top5.reset()

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            discrete_alphas = utils.discretize(
                population.get_population()[step % population.get_population_size()].arch_parameters, device)
            model.copy_arch_parameters(discrete_alphas)
            n = inputs.size(0)
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            population.get_population()[step % population.get_population_size()].objs.update(loss.data, n)
            population.get_population()[step % population.get_population_size()].top1.update(prec1.data, n)
            population.get_population()[step % population.get_population_size()].top5.update(prec5.data, n)

    population.pop_sort()
    for i in range(args.pop_size):
        discrete_alphas = utils.discretize(population.get_population()[i].arch_parameters, device)
        model.copy_arch_parameters(discrete_alphas)
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
        logging.info(
            "[{} Generation] {}/{} finished with validation loss: {}, prec1: {}, prec5: {}, scaling factor: {}, encoding: {}".format(
                gen, i + 1, len(population.get_population()),
                population.get_population()[i].objs.avg,
                population.get_population()[i].top1.avg,
                population.get_population()[i].top5.avg,
                population.get_population()[i].get_mutate_factor(),
                discrete_alphas))


import torch

def fitness_function(x_population, model, valid_queue, criterion, device, max_batches=2):
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
                #print(f"Processing arch_parameters {i}, size: {arch_parameters.size()}")

                if arch_parameters.nelement() != 112:
                    reshaped = False
                    for dim_size in arch_parameters.shape:
                        if dim_size == 112:
                            arch_parameters = arch_parameters.view(-1, 112)[0]
                            reshaped = True
                            break

                    if not reshaped:
                        print(f"Error: Unable to reshape arch_parameters {i}. Current shape: {arch_parameters.size()}")
                        continue
                else:
                    arch_parameters = arch_parameters.view(112)

                reshaped_params = arch_parameters.view(14, 8)
                #print(f"Reshaped arch_parameters {i} to size: {reshaped_params.size()}")

                reshaped_params = reshaped_params.to(device)
                #rint(f"Reshaped params device: {reshaped_params.device}")
                model.copy_arch_parameters(reshaped_params)

                #print(f"Model input size: {inputs.size()}")
                logits = model(inputs)

                loss = criterion(logits, targets)
                #print(f"Calculated loss: {loss.item()}")

                batch_fitness_values.append(-loss.item())

            if batch_fitness_values:
                fitness_values.append(batch_fitness_values)

        if fitness_values:
            # 平均每个 batch 的 fitness 并展平成一维张量
            fitness_summarized = torch.tensor([sum(f) / max_batches for f in zip(*fitness_values)],
                                              dtype=torch.float32)
        else:
            print("No valid fitness values were calculated.")
            fitness_summarized = torch.tensor([], dtype=torch.float32)

    return fitness_summarized





# Initialize
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device("cuda:{}".format(args.gpu))

torch.cuda.set_device(args.gpu)
cudnn.deterministic = True
cudnn.enabled = True
cudnn.benchmark = False

CIFAR_CLASSES = 10
criterion = nn.CrossEntropyLoss().to(device)
model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, device)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                            weight_decay=args.weight_decay)

train_transform, valid_transform = utils._data_transforms_cifar10(args)
train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(args.train_portion * num_train))

train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, pin_memory=False, num_workers=2,
                                          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]))
valid_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
    pin_memory=False, num_workers=2)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

# Create initial population
population = Population(args.pop_size, model._steps, device)

DIR = "3-diffevo_search10-gmm-{}".format(time.strftime("%Y%m%d-%H%M%S"))
if args.dir is not None:
    if not os.path.exists(args.dir):
        utils.create_exp_dir(args.dir)
    DIR = os.path.join(args.dir, DIR)
else:
    DIR = os.path.join(os.getcwd(), DIR)
utils.create_exp_dir(DIR)
utils.create_exp_dir(os.path.join(DIR, "weights"))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(DIR, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("gpu device = {}".format(args.gpu))
logging.info("args =  %s", args)

writer = SummaryWriter(os.path.join(DIR, 'runs', 'de'))

# STAGE 1
start = time.time()
for epoch in range(args.warm_up):
    start_time = time.time()
    logging.info('[INFO] Warming up!!!')
    warm_train(model, train_queue, criterion, optimizer)
    logging.info("[INFO] Warming up finished in {} minutes".format((time.time() - start_time) / 60))

# Instantiate DiffEvo
de_optimizer = DiffEvo(num_step=3, density='gmm', noise=1.0, scaling=1, kde_bandwidth=0.1)

# 获取使用的密度算法并输出
density_method = de_optimizer.get_density_method()
output_message = f"{density_method} was used for density estimation."

# 打印和记录
writer.add_text('Density Estimator', output_message)
print(output_message)
# STAGE 1

for epoch in range(args.epochs):
    logging.info("[INFO] Generation {} training with learning rate {}".format(epoch + 1, scheduler.get_lr()[0]))
    start_time = time.time()

    train(model, train_queue, criterion, optimizer, epoch + 1)
    logging.info("[INFO] Training finished in {} minutes".format((time.time()- start_time) / 60))
    torch.save(model.state_dict(), "cifar10-diffevo-3-gmm-model.pt")
    scheduler.step()

    logging.info("[INFO] Evaluating Generation {} ".format(epoch + 1))
    validation(model, valid_queue, criterion, epoch + 1, population)


    # DiffEvo optimization
    class Individual:
        def __init__(self, arch_params):
            self.arch_parameters = arch_params
        def set_arch_parameters(self, alphas_normal, alphas_reduce):
            self.alphas_normal = alphas_normal
            self.alphas_reduce = alphas_reduce


    # 获取架构参数并转换为 NumPy 数组
    arch_parameters = population.get_arch_parameters()

    # 确保所有 arch_parameters 是 torch.Tensor 并且把它们转移到 CPU 并转换为 numpy 数组
    arch_parameters_cpu = [p.cpu().numpy() for p in arch_parameters]

    initial_population_list = []

    # 构建初始种群并确保每个 arch_parameters 被转换为一维的 NumPy 数组
    for idx, params in enumerate(arch_parameters_cpu):
        if isinstance(params, np.ndarray):
            params_flat = params.flatten()  # 使其展平为一维
        else:
            raise ValueError(f"Expected params to be numpy array, but got {type(params)}")
        initial_population_list.append(params_flat)

        #print(f"Index {idx}, Flattened arch_parameters shape: {params_flat.shape}")

    # 转换为 PyTorch Tensor
    x = torch.tensor(initial_population_list, dtype=torch.float32)
    #print(f"Converted tensor shape: {x.shape}")

    x_flattened_reshaped = x.view(x.size(0), -1)
    #print(f"Flattened reshaped tensor shape: {x_flattened_reshaped.shape}")

    # 执行 DiffEvo 优化
    fitness_with_args = partial(fitness_function, model=model, valid_queue=valid_queue, criterion=criterion,
                                device=device)
    # 检查在 DiffEvo 优化调用前的尺寸

    optimized_population_objects = de_optimizer.optimize(fitness_with_args, x_flattened_reshaped, trace=False)
    def process_optimized_population(optimized_population_np):
        num_individuals = optimized_population_np.shape[0] // 2  # 使用前一半
        transformed_population = []
        for i in range(num_individuals):
            part_normal_all_elements = optimized_population_np[2 * i].reshape(-1)[:112]
            part_reduce_all_elements = optimized_population_np[2 * i + 1].reshape(-1)[:112]
            part_normal = part_normal_all_elements.reshape(14, 8)
            part_reduce = part_reduce_all_elements.reshape(14, 8)
            transformed_population.append([part_normal, part_reduce])
        return np.array(transformed_population)

    # 从优化后的对象中提取参数，并转为 NumPy 数组以更新回 population 对象
    optimized_population_np = optimized_population_objects.cpu().numpy()
    print(f"optimized_population_np shape: {optimized_population_np.shape}")
    processed_population = process_optimized_population(optimized_population_np)
    print(f"processed_populationshape: {processed_population.shape}")
    population_instance = Population(args.pop_size, model._steps, device)
    population_instance.set_arch_parameters(processed_population)


    # 记录评价指标
    for i, p in enumerate(population.get_population()):
        writer.add_scalar("pop_top1_{}".format(i + 1), p.get_fitness(), epoch + 1)
        writer.add_scalar("pop_top5_{}".format(i + 1), p.top5.avg, epoch + 1)
        writer.add_scalar("pop_obj_valid_{}".format(i + 1), p.objs.avg, epoch + 1)

    # 保存population信息
    with open(os.path.join(DIR, "population_{}.pickle".format(epoch + 1)), 'wb') as f:
        pickle.dump(population, f)

    last = time.time() - start_time
    logging.info("[INFO] {}/{} epoch finished in {} minutes".format(epoch + 1, args.epochs, last / 60))
    utils.save(model, os.path.join(DIR, "weights", "weights.pt"))

writer.close()
last = time.time() - start
logging.info("[INFO] Total search time: {} hours".format(last / 3600))