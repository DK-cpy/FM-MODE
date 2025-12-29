import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
#from model_search import Network
import pickle
import population
import genotypes
import torch.nn.functional as F

class AvgrageMeter(object):

	def __init__(self):
		self.reset()

	def reset(self):
		self.avg = 0
		self.sum = 0
		self.cnt = 0

	def update(self, val, n=1):
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		#correct_k = correct[:k].view(-1).float().sum(0)
		correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0/batch_size))
	return res
'''
def accuracy2(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)
	print("[INFO] topk: {}, maxk: {}, batch_size: {}".format(topk, maxk, batch_size))

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0/batch_size))
	return res
'''
class Cutout(object):
	def __init__(self, length):
		self.length = length

	def __call__(self, img):
		h, w = img.size(1), img.size(2)
		mask = np.ones((h, w), np.float32)
		y = np.random.randint(h)
		x = np.random.randint(w)

		y1 = np.clip(y - self.length // 2, 0, h)
		y2 = np.clip(y + self.length // 2, 0, h)
		x1 = np.clip(x - self.length // 2, 0, w)
		x2 = np.clip(x + self.length // 2, 0, w)

		mask[y1: y2, x1: x2] = 0.
		mask = torch.from_numpy(mask)
		mask = mask.expand_as(img)
		img *= mask
		return img


def _data_transforms_cifar10(params):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if isinstance(params, dict) and params.get('cutout', False):
        train_transform.transforms.append(Cutout(params.get('cutout_length', 0)))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform

def _data_transforms_cifar100(args):
	CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
	CIFAR_STD = [0.2675, 0.2565, 0.2761]

	train_transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
	])
	if args.cutout:
		train_transform.transforms.append(Cutout(args.cutout_length))

	valid_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
		])
	return train_transform, valid_transform

def count_parameters_in_MB(model):
	return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
	filename = os.path.join(save, 'checkpoint.pth.tar')
	torch.save(state, filename)
	if is_best:
		best_filename = os.path.join(save, 'model_best.pth.tar')
		shutil.copyfile(filename, best_filename)


def save(model, model_path):
	torch.save(model.state_dict(), model_path)


def load(model, model_path, gpu = 0):
	model.load_state_dict(torch.load(model_path, map_location = 'cuda:{}'.format(gpu)), strict=False)

def drop_path(x, drop_prob):
	if drop_prob > 0.:
		keep_prob = 1.-drop_prob
		mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
		x.div_(keep_prob)
		x.mul_(mask)
	return x


def create_exp_dir(path, scripts_to_save=None):
	if not os.path.exists(path):
		os.mkdir(path)
	print('Experiment dir : {}'.format(path))

	if scripts_to_save is not None:
		os.mkdir(os.path.join(path, 'scripts'))
		for script in scripts_to_save:
			dst_file = os.path.join(path, 'scripts', os.path.basename(script))
			shutil.copyfile(script, dst_file)

def check_equality(model, parameters):
	'''
	if model.arch_parameters[0].device != parameters[0].device:
		tmp = []
		tmp.append(torch.zeros_like(parameters[0]))
		tmp.append(torch.zeros_like(parameters[1]))

		for x, y in zip(tmp, paramerters):
			x.data.copy_(y.data)
	'''
	for x, y in zip(model.arch_parameters(), parameters):
		if not torch.all(x.eq(y)):
			return False
	return True

'''
def model_copy(model, criterion, device):
	model_copy = Network(model._C, model._num_classes, model._layers, criterion, device)
	#model_cpu = Network(model._C, model._num_classes, model._layers, criterion, device = torch.device("cpu"))
	
	#model_cpu.load_state_dict(model.state_dict())
	model_copy.load_state_dict(model.state_dict())
	model_copy.to(device)
	for x, y in zip(model_copy.arch_parameters(), model.arch_parameters()):
		x.data.copy_(y.data)

	return model_copy
'''

def load_population(file_name):
	pop = population.Population(0, 0)
	with open(file_name, 'rb') as f:
		pop = pickle.load(f)
		#pop = torch.load(f, map_location = torch.device('cuda:0'))
	return pop

def derive_architecture(alphas):
	genotype = genotypes.PRIMITIVES
	ops = []
	for alpha in alphas:
		tmp = F.softmax(alpha, dim = -1)
		index = torch.argmax(tmp)
		#op = genotype[torch.argmax(tmp)]
		op = genotype[index]
		if op == 'none':
			index = torch.topk(tmp, 2).indices[1]
			op = genotype[index]
		#print("arch: {}, ops{}: {}".format(tmp, index, op))
		ops.append((op, tmp[index]))

	return ops


def derive_architecture_topk(alphas, k=2):
	genotype = genotypes.PRIMITIVES
	num = alphas.size(0)
	#print(f"Number of elements in alpha (num): {num}")
	#print("Alphas content at start of derive_architecture_topk:", alphas)

	steps = None
	i = 1
	while i <= num:
		tmp = ((i + 1) * (i + 2)) / 2
		if ((tmp - 1) == num):
			steps = i
			#print("steps: {}".format(steps))
			break
		i += 1

	if steps is None:
		raise ValueError(
			f"Failed to determine steps for num: {num}, the number of elements in alpha might be incorrect.")

	ops = []
	offset = 0
	for i in range(steps):
		t = []
		for j in range(2 + i):
			tmp = F.softmax(alphas[offset + j], dim=-1)
			index = torch.argmax(tmp)
			op = genotype[index]
			if op == 'none':
				index = torch.topk(tmp, 2).indices[1]
				op = genotype[index]
			t.append((op, j, tmp[index]))

		offset += j + 1
		t.sort(key=lambda x: x[2], reverse=True)
		for ip in range(k):
			p = t[ip]
			ops.append(tuple([p[0], p[1]]))
	return ops


def discretize(alphas, device):
	genotype = genotypes.PRIMITIVES
	#print("alphas at start of discretize: ", alphas)

	# Ensure alphas is a list with two elements
	if not isinstance(alphas, list) or len(alphas) != 2:
		raise ValueError("alphas should be a list with exactly two elements.")

	normal_alphas, reduction_alphas = alphas[0], alphas[1]
	normal_cell = derive_architecture_topk(normal_alphas)
	reduction_cell = derive_architecture_topk(reduction_alphas)
	#print(f"Generated normal_cell: {normal_cell}")
	#print(f"Generated reduction_cell: {reduction_cell}")

	num = normal_alphas.size(0)
	#print("num_discretize: ", num)

	steps = None
	i = 1
	while i <= num:
		tmp = ((i + 1) * (i + 2)) / 2
		if ((tmp - 1) == num):
			steps = i
			break
		i += 1

	if steps is None:
		raise ValueError("Unable to determine 'steps' from the given alphas.")

	new_normal = torch.zeros_like(normal_alphas).to(device)
	new_reduction = torch.zeros_like(reduction_alphas).to(device)

	# Filling new_normal
	offset = 0
	for i in range(len(normal_cell) // 2):
		for j in range(2):
			op, cell = normal_cell[2 * i + j]
			index = genotype.index(op)
			if 0 <= offset + cell < new_normal.size(0):
				new_normal[offset + cell][index] = 1
		offset += (i + 2)

	# Filling new_reduction
	offset = 0
	for i in range(len(reduction_cell) // 2):
		for j in range(2):
			op, cell = reduction_cell[2 * i + j]
			index = genotype.index(op)
			if 0 <= offset + cell < new_reduction.size(0):
				new_reduction[offset + cell][index] = 1
		offset += (i + 2)

	new_alphas = [new_normal, new_reduction]
	return new_alphas


def discretize2(alphas, beta, device):
	genotype = genotypes.PRIMITIVES
	normal_cell = derive_architecture_topk(alphas[0])
	reduction_cell = derive_architecture_topk(alphas[1])
	#print("Normal: ", normal_cell)
	#print("Reduction: ", reduction_cell)

	num = alphas[0].size()[0]
	# To get the value of steps from alphas
	i = 1
	while i <= num:
		tmp = ((i + 1) * (i + 2)) / 2
		if ((tmp - 1) == num):
			steps = i
			break
		i += 1

	new_alphas = []
	new_normal = torch.zeros_like(alphas[0]).to(device)
	new_reduction = torch.zeros_like(alphas[1]).to(device)
	
	i = 0
	offset = 0
	while i < len(normal_cell):
		op, cell = normal_cell[i]
		index = genotype.index(op)
		#print("i: {}, offset: {}, cell: {}".format(i, offset, cell))
		#print("op: {}, alphas[0][offset + cell]: {}, index: {}".format(op, alphas[0][int(offset + cell)], index))
		new_normal[int(offset + cell)][index] = beta
		#print("new_normal[offset + cell]: {}, index: {}".format(new_normal[int(offset + cell)], index))
		
		i += 1
		op, cell = normal_cell[i]
		index = genotype.index(op)
		#print("i: {}, offset: {}, cell: {}".format(i, offset, cell))
		#print("op: {}, alphas[0][offset + cell]: {}, index: {}".format(op, alphas[0][int(offset + cell)], index))
		new_normal[int(offset + cell)][index] = beta
		#print("new_normal[offset + cell]: {}, index: {}".format(new_normal[int(offset + cell)], index))
		offset += (i // 2) + 2
		i += 1

	i = 0
	offset = 0
	while i < len(reduction_cell):
		op, cell = reduction_cell[i]
		index = genotype.index(op)
		#print("i: {}, offset: {}, cell: {}".format(i, offset, cell))
		#print("op: {}, alphas[1][offset + cell]: {}, index: {}".format(op, alphas[1][int(offset + cell)], index))
		new_reduction[int(offset + cell)][index] = beta
		#print("new_reduction[offset + cell]: {}, index: {}".format(new_reduction[int(offset + cell)], index))
		
		i += 1
		op, cell = reduction_cell[i]
		index = genotype.index(op)
		#print("i: {}, offset: {}, cell: {}".format(i, offset, cell))
		#print("op: {}, alphas[1][offset + cell]: {}, index: {}".format(op, alphas[1][int(offset + cell)], index))
		new_reduction[int(offset + cell)][index] = beta
		#print("new_reduction[offset + cell]: {}, index: {}".format(new_reduction[int(offset + cell)], index))
		offset += (i // 2) + 2
		i += 1
	
	new_alphas = [new_normal, new_reduction]
	#print('new_normal: ', derive_architecture_topk(new_normal))
	#print('new_reduction: ', derive_architecture_topk(new_reduction))
	#assert normal_cell == derive_architecture_topk(new_normal)
	#assert reduction_cell == derive_architecture_topk(new_reduction)
	return new_alphas

























