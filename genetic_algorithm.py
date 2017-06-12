# -*- coding: utf-8 -*-
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import yaml

from copy import copy
from pprint import pprint
from time import time

class Individual(object):
    '''
    个体类
    '''
    def __init__(self, *args, **kwargs):
        length = args[0]
        x = []
        for _ in range(length):
            if random.random() > 0.5: 
                x.append('1')
            else:
                x.append('0')
        self.chromosome = x
        self.chromosome_str = ''.join(self.chromosome)
        self.object_value = 0.0
        self.object_fitness = 0.0
        

class Population(object):
    '''
    种群类
    '''
    def __init__(self, *args, **kwargs):
        self.population_capacity = args[0]
        self.params = []
        self.chromosome_length = 0
        self.individuals = []
        self.read_parameters('default.yaml')

    def read_parameters(self, configuration_file='parameters.yaml'):
        '''
        读取配置文件
        '''
        with open(configuration_file, 'r') as fd:
            _yaml = yaml.load(fd)
            params = _yaml['params']
            for param in params:
                for k, v in param.items():
                    length = math.ceil(math.log2(int((v[1] - v[0]) / v[2])))
                    self.params.append((v[0], v[1], length))
                    self.chromosome_length += length
        
    def initialize_population(self):
        '''
        初始化种群
        '''
        self.individuals = [Individual(self.chromosome_length) for _ in range(self.population_capacity)]

    def evaluate_object_value(self):
        for individual in self.individuals:
            start = 0
            end = 0

            x = []
            for param in self.params:
                end += param[2]
                # ***** core compute *****
                partial_chromosome = individual.chromosome[start : end]
                value = 0
                for i in range(len(partial_chromosome), 0, -1):
                    shift = len(partial_chromosome) - i
                    t = 1
                    t <<= shift
                    value += (ord(individual.chromosome[i - 1]) - ord('0')) * t
                value = (param[1] - param[0]) * (value / math.pow(2, param[2])) + param[0]
                x.append(value)
                # ***** core compute *****
                start += param[2]

            # ***** core compute *****
            individual.object_value = 100 * (x[0] * x[0] - x[1]) * (x[0] * x[0] - x[1]) + (1 - x[1]) * (1 - x[1])
            # ***** core compute *****

    def evaluate_object_fitness(self):
        for individual in self.individuals:
            individual.object_fitness = individual.object_value


class GeneticAlgorithm(object):
    '''
    遗传算法类
    '''
    def __init__(self, *args, **kwargs):
        self.population_capacity = 0  # 种群容量
        self.e = 0                    # 当前进化代数
        self.epochs = 0               # 最大进化代数
        self.pc = 0.0                 # 交叉概率
        self.px = 0.0                 # 编译概率
        self.read_parameters('default.yaml')

    def read_parameters(self, configuration_file='parameters.yaml'):
        '''
        读取配置文件
        '''
        with open(configuration_file, 'r') as fd:
            _yaml = yaml.load(fd)
            # 遗传参数赋值
            self.population_capacity = _yaml['population_capacity']
            self.epochs = _yaml['epochs']
            self.pc = _yaml['pc']
            self.px = _yaml['px']

    def run(self):
        '''
        主循环
        '''
        self.population = Population(self.population_capacity)
        self.population.initialize_population()
        self.current_best_individual = Individual(self.population.chromosome_length)
        self.current_worst_individual = Individual(self.population.chromosome_length)
        self.current_worst_individual_index = 0
        self.best_individual = Individual(self.population.chromosome_length)
    
        while self.e < self.epochs:
            self.estimate()
            self.generate_next_population()
            self.make_visualization(self.e + 1, self.best_individual.object_value)
            self.e += 1
    
    def estimate(self):
        '''
        个体评估
        '''
        self.evaluate_object_value()
        self.evaluate_object_fitness()
        self.select_best_and_worst_individual()

    def evaluate_object_value(self):
        '''
        计算目标值
        '''
        self.population.evaluate_object_value()

    def evaluate_object_fitness(self):
        '''
        计算适应度值
        '''
        self.population.evaluate_object_fitness()

    def select_best_and_worst_individual(self):
        '''
        筛选当前最优和最差个体，更新全局最优个体
        '''
        self.current_best_individual = copy(self.population.individuals[0])
        self.current_worst_individual = copy(self.population.individuals[0])
        self.current_worst_individual_index = 0
        for i in range(1, self.population_capacity):
            if self.population.individuals[i].object_fitness > self.current_best_individual.object_fitness:
                self.current_best_individual = copy(self.population.individuals[i])
            elif self.population.individuals[i].object_fitness < self.current_worst_individual.object_fitness:
                self.current_worst_individual = copy(self.population.individuals[i])
                self.current_worst_individual_index = i

        if self.e == 0:
            self.best_individual = copy(self.current_best_individual)
        else:
            if self.current_best_individual.object_fitness > self.best_individual.object_fitness:
                self.best_individual = copy(self.current_best_individual)
        
        self.improve_evolution()
    
    def improve_evolution(self):
        '''
        改善进化
        '''
        self.population.individuals[self.current_worst_individual_index] = copy(self.best_individual)

    def generate_next_population(self):
        self.select_operator()
        self.crossover_operator()
        self.mutate_operator()

    def select_operator(self):
        '''
        选择算子－比例选择，可选方法：最优保存策略、确定式采样选择、无回放随机选择、无回放余数随机选择、排序选择、随机联赛选择
        '''
        _sum = 0.0
        for i in range(self.population_capacity):
            _sum += self.population.individuals[i].object_fitness
        _cum = [self.population.individuals[i].object_fitness / _sum for i in range(self.population_capacity)]
        for i in range(1, self.population_capacity):
            _cum[i] = _cum[i] + _cum[i - 1]

        for i in range(self.population_capacity):
            count = 0
            while (random.random() > _cum[i] and count < self.population_capacity):
                count += 1
            if count < self.population_capacity:
                self.population.individuals[i] = copy(self.population.individuals[count])

    def crossover_operator(self):
        '''
        交叉算子－单点交叉，可选方法：多点交叉、均匀交叉、算数交叉
        '''
        index = np.empty(shape=(self.population_capacity, ), dtype=np.int64)
        for i in range(self.population_capacity):
            index[i] = i
        np.random.shuffle(index)

        for i in range(0, self.population_capacity, 2):
            if (random.random() < self.pc):
                crossover_point = random.randint(1, self.population.chromosome_length)
                for j in range(crossover_point, self.population.chromosome_length):
                    self.population.individuals[i].chromosome[j], self.population.individuals[i + 1].chromosome[j] =\
                    self.population.individuals[i + 1].chromosome[j], self.population.individuals[i].chromosome[j]

    def mutate_operator(self):
        '''
        变异算子－基本位变异，可选方法：均匀变异，边界变异，非均匀变异，高斯变异
        '''
        for i in range(self.population_capacity):
            for j in range(self.population.chromosome_length):
                if (random.random() < self.px):
                    if self.population.individuals[i].chromosome[j] == '0':
                        self.population.individuals[i].chromosome[j] == '1'
                    else:
                        self.population.individuals[i].chromosome[j] == '0'
    
    def make_visualization(self, x, y):
        print(f"[+] epoch - {x} : current best value is \033[1;31;40m{y}\033[0m ...")

if __name__ == '__main__':
    ga = GeneticAlgorithm()
    ga.run()
