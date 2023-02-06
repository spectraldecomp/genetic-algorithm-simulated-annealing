import random
from ga_util import bitstr2float
import numpy as np





class genetic_algorithm:

    def __init__(self, population_size, individual_size, individual_max, individual_min, mutate_probability, crossover_probability, fit_func, is_min):
        self.population_size = population_size
        self.individual_size = individual_size
        self.individual_max = individual_max
        self.individual_min = individual_min
        self.mutate_probability = mutate_probability
        self.crossover_probability = crossover_probability
        self.fit_func = fit_func
        self.is_min = is_min


    def generate_population(self, population_size, individual_size, individual_min, individual_max):
        # Creates initial population, values chosen uniformly from valid range
        population = [[0 for _ in range(individual_size)] for _ in range(population_size)]
        for individual in range(population_size):
            for index in range(individual_size):
                population[individual][index] = random.uniform(individual_min, individual_max)
        return population

    def select_parents(self, population, elite_size):
        # parents selected with elitism
        fitness_scores = [self.fit_func(np.array(individual)) for individual in population]
        population_with_fitness = list(zip(population, fitness_scores))
        if self.is_min:
            population_with_fitness.sort(key=lambda x: x[1])
        else:
            population_with_fitness.sort(key=lambda x: x[1], reverse=True)
        elite = [individual for individual, score in population_with_fitness[:elite_size]]
        parents = elite[:]
        while len(parents) < len(population):
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)
            parents.append(parent1)
            parents.append(parent2)
        return parents

    def crossover(self, parents, children_size):
        # children are generates if value generated less than crossover rate. Single point crossover
        children = []
        for i in range(children_size):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            if random.uniform(0, 1) <self.crossover_probability:
                children.append(parent1[:len(parent1)//2] + parent2[len(parent2)//2:])
        return children

    def mutate(self, population, mutation_probability):
        # Randomly changes value of chromosome if less than mutation probability
        for i, individual in enumerate(population):
            for j, gene in enumerate(individual):
                if random.uniform(0, 1) < mutation_probability:
                    population[i][j] = random.uniform(self.individual_min, self.individual_max)
        return population

    def genetic_algorithm_run(self, population_size, individual_size, elite_size, offspring_size, mutation_probability, generations):
        #main function
        population = self.generate_population(population_size, individual_size, self.individual_min, self.individual_max)
        for i in range(generations):
            parents = self.select_parents(population, elite_size)
            offspring = self.crossover(parents, offspring_size)
            population = self.mutate(offspring, mutation_probability)
        if self.is_min:
            minimum = population[0]
            for pair in population:
               if self.fit_func(pair) < self.fit_func(minimum):
                   minimum = pair
            return minimum
        else:
            maximum = population[0]
            for pair in population:
                if self.fit_func(pair) > self.fit_func(maximum):
                    maximum = pair
            return maximum