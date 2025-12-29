"""GanAlgorithm."""
from __future__ import absolute_import, division, print_function
import copy
import os
import numpy as np
import random
from tqdm import tqdm
from pyswarm import pso

# To implement easy, wo use int number in the code to represent the
# corresponding operations. These are replaced with one-hot in the paper,
# which can better explain our theory. Each int number is equal to a one-hot number.


class GanAlgorithm():

    def __init__(self, args):
        self.steps = 3
        self.up_nodes = 2
        self.down_nodes = 2
        self.normal_nodes = 5
        self.dis_normal_nodes = 5
        self.archs = {}
        self.dis_archs = {}
        self.base_dis_arch = np.array(
            [[3, 0, 3, 0, 2, 0, 0], [3, 0, 3, 0, 1, -1, -1], [3, 0, 3, 0, 1, -1, -1]])
        self.Normal_G = []
        self.Up_G = []
        self.Normal_G_fixed = []
        self.Up_G_fixed = []
        for i in range(self.steps * self.normal_nodes):
            self.Normal_G.append([0, 1, 2, 3, 4, 5, 6])
        for i in range(self.steps * self.up_nodes):
            self.Up_G.append([0, 1, 2])
        for i in range(self.steps * self.normal_nodes):
            self.Normal_G_fixed.append([0, 1, 2, 3, 4, 5, 6])  # ([0,1,2,3,4,5,6])
        for i in range(self.steps * self.up_nodes):
            self.Up_G_fixed.append([0, 1, 2])
    def global_pso_optimize(self, population, fid_stat, trainer):
        """
        对整个种群执行PSO优化
        :param population: 当前种群（numpy数组）
        :param fid_stat: FID统计量
        :param trainer: GenTrainer实例（用于访问评估方法）
        :return: 优化后的种群
        """
        optimized_population = []
        
        for genotype in tqdm(population, desc="Global PSO Optimization"):
            def fitness(x):
                # 添加安全约束
                candidate = np.clip(x.reshape(3,7), 0, 6)  # 先约束范围
                candidate = candidate.astype(int)  # 再转换类型
                
                # 检查有效性并施加惩罚
                invalid_mask = (candidate < 0) | (candidate > 6)
                if np.any(invalid_mask):
                    return float('inf')  # 极大惩罚无效解
                    
                try:
                    is_value, _, fid_value = trainer.validate(candidate, fid_stat)
                except IndexError as e:
                    print(f"Invalid genotype: {candidate}")
                    return float('inf')  # 异常时返回极大值
                    
                return fid_value - is_value
            
            # 修改边界生成方式（保留1单位变化空间）
            base = genotype.flatten()
            lb = np.clip(base - 1, 0, 6)
            ub = np.clip(base + 1, 0, 6)
            
            # 执行带约束的PSO
            best_pos, _ = pso(fitness, lb, ub, 
                             swarmsize=15,
                             maxiter=10,
                             phip=1.2,
                             phig=1.8,
                             debug=False)
                             
            # 最终安全转换
            optimized = np.clip(best_pos.reshape(3,7), 0, 6).astype(int)
            optimized_population.append(optimized)
            
            # 打印调试信息
            print(f"Original: {genotype}")
            print(f"Optimized: {optimized}\n")
        
        return np.array(optimized_population)
    def pso_refine(self, base_genotype):
        # 粒子群优化微调，局部PSO方法
        def fitness(x):
            # 转换基因型并评估
            return self.evaluate_genotype(x)

        lb = base_genotype.flatten() - 1
        ub = base_genotype.flatten() + 1
        lb = np.clip(lb, 0, 6)
        ub = np.clip(ub, 0, 6)
        
        best_pos, _ = pso(fitness, lb, ub)
        return best_pos.reshape(3,7).astype(int)
    def search(self, remove=True):
        # 保留原有遗传算法采样
        if np.random.rand() < 0.7:  # 70%概率使用遗传算法
            new_genotype = self.sample_fair(remove)
        else:  # 30%概率使用PSO优化
            base = self.sample_fair(remove)
            new_genotype = self.pso_refine(base)
        return new_genotype
    
    
    def Get_Operation(self, Candidate_Operation, mode, num, remove=True):
        '''
        Candidate_Operation: operation pool
        mode: Operation type
        return: the operation
        remove：remove from the pool
        '''
        if Candidate_Operation == [] and mode == 'up':
            Candidate_Operation += self.Up_G_fixed[num]
        elif Candidate_Operation == [] and mode == 'normal':
            Candidate_Operation += self.Normal_G_fixed[num]
        choice = random.choice(Candidate_Operation)
        if remove:
            Candidate_Operation.remove(choice)
        return choice

    def uniform_sample(self):
        genotype = np.zeros(
            (self.steps, self.up_nodes + self.normal_nodes), dtype=int)
        for i in range(self.steps):
            for j in range(self.up_nodes):
                genotype[i][j] = random.randint(0, 2)
            while (genotype[i][2] == 0 and genotype[i][3] == 0):
                for k in range(2):
                    genotype[i][k + 2] = random.randint(0, 6)
            while (genotype[i][4] == 0 and genotype[i][5] == 0 and genotype[i][6] == 0):
                for k in range(2, self.normal_nodes):
                    genotype[i][k + 2] = random.randint(0, 6)
        return genotype

    def sample_fair(self, remove=True):
        genotype = np.zeros(
            (self.steps, self.up_nodes + self.normal_nodes), dtype=int)  # 3*7
        for i in range(self.steps):
            for j in range(self.up_nodes):
                genotype[i][j] = self.Get_Operation(self.Up_G[2 * i + j], 'up', 2 * i + j, remove)

            for k in range(2):
                genotype[i][k + 2] = self.Get_Operation(self.Normal_G[5 * i + k], 'normal', 5 * i + k, remove)
            while (genotype[i][2] == 0 and genotype[i][3] == 0):
                for k in range(2):
                    genotype[i][k + 2] = self.Get_Operation(self.Normal_G[5 * i + k], 'normal', 5 * i + k, False)

            for k in range(2, self.normal_nodes):
                genotype[i][k + 2] = self.Get_Operation(self.Normal_G[5 * i + k], 'normal', 5 * i + k, remove)
            while (genotype[i][4] == 0 and genotype[i][5] == 0 and genotype[i][6] == 0):
                for k in range(2, self.normal_nodes):
                    genotype[i][k + 2] = self.Get_Operation(self.Normal_G[5 * i + k], 'normal', 5 * i + k, False)

        return genotype

    def sample_zero(self, remove=True):
        genotype = np.zeros(
            (self.steps, self.up_nodes + self.normal_nodes), dtype=int)
        for i in range(self.steps):
            for j in range(self.up_nodes):
                genotype[i][j] = self.Get_Operation(self.Up_G[2 * i + j], 'up', 2 * i + j, remove)
        return genotype

    def encode(self, genotype):
        lists = [0 for i in range(self.steps)]
        for i in range(len(lists)):
            lists[i] = str(genotype[i])
        return tuple(lists)

    def search(self, remove=True):
        new_genotype = self.sample_fair(remove)
        # new_genotype = self.uniform_sample()
        return new_genotype

    def sample_D(self):
        genotype = np.zeros((self.steps, self.down_nodes +
                             self.normal_nodes), dtype=int)
        for i in range(self.steps):
            genotype[i][0] = random.randint(1, 6)
            while (genotype[i][1] == 0 and genotype[i][2] == 0):
                for k in range(2):
                    genotype[i][k + 2] = random.randint(0, 6)
            while (genotype[i][3] == 0 and genotype[i][4] == 0):
                for k in range(2, 4):
                    genotype[i][k + 1] = random.randint(0, 6)
            genotype[i][self.dis_normal_nodes] = random.randint(-1, 5)
            if genotype[i][self.dis_normal_nodes] == -1:
                genotype[i][self.dis_normal_nodes + 1] = -1
            else:
                genotype[i][self.dis_normal_nodes + 1] = random.randint(0, 5)
        return genotype

    def judge_repeat(self, new_genotype):
        t = self.encode(new_genotype)
        return t in self.archs

    def judge_repeat_dis(self, new_genotype):
        t = self.encode(new_genotype)
        return t in self.dis_archs

    def search_mutate_dis(self):
        t = self.encode(self.base_dis_arch)
        self.dis_archs[t] = self.base_dis_arch
        while (t in self.dis_archs):
            new_genotype = self.mutation_gen(self.base_dis_arch)
            t = self.encode(new_genotype)
        self.dis_archs[t] = new_genotype
        return new_genotype

    def mutation_dis(self, alphas_a, ratio=0.5):
        """Mutation for an individual"""
        new_alphas = alphas_a.copy()
        layer = random.randint(0, self.steps - 1)
        index = random.randint(0, self.down_nodes + self.dis_normal_nodes - 1)
        if index == 0:
            new_alphas[layer][index] = random.randint(1, 6)
            while (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(1, 6)
        elif index >= 1 and index < 3:
            new_alphas[layer][index] = random.randint(0, 6)
            while (new_alphas[layer][1] == 0 and new_alphas[layer][2] == 0) or (
                    new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        elif index >= 3 and index < 5:
            new_alphas[layer][index] = random.randint(0, 6)
            while (new_alphas[layer][3] == 0 and new_alphas[layer][4] == 0) or (
                    new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        if index == 5:
            new_alphas[layer][index] = random.randint(-1, 5)
            while (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(-1, 5)
        if index == 6:
            new_alphas[layer][index] = random.randint(0, 5)
            while (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 5)
        return new_alphas

    def mutation_gen(self, alphas_a, ratio=0.5):
        """Mutation for an individual"""
        new_alphas = alphas_a.copy()
        layer = random.randint(0, self.steps - 1)
        index = random.randint(0, self.down_nodes + self.normal_nodes - 1)
        if index < 2:
            new_alphas[layer][index] = random.randint(0, 2)
            while (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 2)
        elif index >= 2 and index < 4:
            new_alphas[layer][index] = random.randint(0, 6)
            while (new_alphas[layer][2] == 0 and new_alphas[layer][3] == 0) or (
                    new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        elif index >= 4:
            new_alphas[layer][index] = random.randint(0, 6)
            while (new_alphas[layer][3] == 0 and new_alphas[layer][4] == 0) or (
                    new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        return new_alphas
