import numpy as np
import random

import numpy as np
import random

class DiffEvoSearchAlgorithm:
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

    def search(self, remove=True):
        new_genotype = self.sample_fair(remove)
        return new_genotype
    def encode(self, genotype):
        """将基因型编码为元组，用于去重"""
        return tuple(str(row) for row in genotype)  # 将每行转换为字符串，组成元组