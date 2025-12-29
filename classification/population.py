import random
import pickle
from chromosomes import *
import logging
import torch

class Population:
    def __init__(self, pop_size, steps, device=torch.device("cpu")):
        self._pop_size = pop_size
        self._device = device
        self._steps = steps
        self.population = [chromosome(steps, self._device) for _ in range(pop_size)]
        #print(f"Population initialized with size: {len(self.population)}")

        if len(self.population) == 0:
            print("Warning: Population is empty after initialization.")

    def __iter__(self):
        return iter(self.population)

    def __len__(self):
        return len(self.population)

    def pop(self, index):
        if 0 <= index < len(self.population):
            return self.population.pop(index)
        else:
            raise IndexError("Index out of range.")

    def __getitem__(self, index):
        return self.population[index]

    def get_population_size(self):
        return len(self.population)

    def get_population(self):
        return self.population

    def print_population(self):
        for p in self.population:
            print(p.get_fitness())

    def pop_sort(self):
        # 检查 self.population 是否为列表
        if isinstance(self.population, list):
            #print("Sorting population based on fitness...")
            for i, p in enumerate(self.population):
                print(f"Before sorting - Individual {i}: Fitness = {p.get_fitness()}")
            self.population.sort(key=lambda x: x.get_fitness(), reverse=True)
            #print("Population sorted.")
            for i, p in enumerate(self.population):
                print(f"After sorting - Individual {i}: Fitness = {p.get_fitness()}")
        else:
            print(f"Error: self.population is a {type(self.population).__name__}, not a list.")

    def random_pop(self):
        self.population = [chromosome(self._steps, self._device) for _ in range(self._pop_size)]

    def get_population_top50(self):
        self.pop_sort()  # 先排序，再保留前50个
        self.population = self.population[:50]
        return self.population

    def pop_pop(self, indices_to_pop):
        print(f"Current population size before removal: {len(self.population)}")

        if not self.population:
            print("Warning: Attempted to remove individuals from an empty population.")
            return

        # 确保至少保留一个个体
        if len(self.population) <= len(indices_to_pop):
            print("Warning: Cannot remove all individuals, keeping one.")
            return

        print(f"Removing individuals at indices: {indices_to_pop}")
        for index in sorted(indices_to_pop, reverse=True):
            if 0 <= index < len(self.population):
                self.population.pop(index)
                print(f"Removed individual at index: {index}")
            else:
                print(f"Index {index} out of range. Skipping.")

        print(f"New population size after removal: {len(self.population)}")

    def save_population(self, file_name):
        with open(file_name, 'wb') as f:
            if not self.population:
                logging.warning("Population is empty during save.")
            else:
                pickle.dump(self.population, f)
                logging.info("Population successfully saved.")

    def load_population(self, file_name):
        with open(file_name, 'rb') as f:
            self.population = pickle.load(f)
            print(f"Population loaded from {file_name}, size: {len(self.population)}")

    def get_arch_parameters(self):
        """Collect architecture parameters from each individual in the population."""
        arch_params = [arch_param for individual in self.population for arch_param in individual.get_arch_parameters()]
        for idx, params in enumerate(arch_params):
            print(f"[DEBUG] get_arch_parameters - Param {idx}: shape = {params.shape}")
        return arch_params
    def set_arch_parameters(self, new_arch_parameters):
        """Update architecture parameters for each individual in the population."""
        for individual, params in zip(self.population, new_arch_parameters):
            alphas_normal = params[0]  # 形状为 (14, 8)
            alphas_reduce = params[1]  # 形状为 (14, 8)
            individual.set_arch_parameters(alphas_normal, alphas_reduce)

    def create_new_individual(self):
        return chromosome(self._steps, self._device)
