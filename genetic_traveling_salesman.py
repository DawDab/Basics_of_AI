from dataclasses import dataclass
from typing import List, Tuple
import random
import logging
import math

@dataclass
class City:
    """Represents a city with x and y coordinates."""
    x: float
    y: float

class GeneticTSP:
    """Genetic algorithm implementation for solving the Traveling Salesman Problem."""
    
    def __init__(self, cities: List[City], 
                 population_size: int = 100, generations: int = 1000,
                 mutation_rate: float = 0.1, tournament_size: int = 5) -> None:
        """
        Initialize the genetic algorithm solver.
        
        Args:
            cities: List of City objects representing the cities
            population_size: Size of the population in each generation
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation for each gene
            tournament_size: Number of chromosomes in tournament selection
        """
        self.cities = cities
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.best_fitness_history = []
        logging.basicConfig(level=logging.INFO)
        
    def create_initial_population(self) -> List[List[int]]:
        """Create random initial population of chromosomes."""
        population = []
        for _ in range(self.population_size):
            chromosome = list(range(len(self.cities)))
            random.shuffle(chromosome)
            population.append(chromosome)
        return population
    
    def calculate_fitness(self, chromosome: List[int]) -> float:
        """Calculate fitness value for a chromosome."""
        total_distance = 0.0
        for i in range(len(chromosome)):
            city1 = self.cities[chromosome[i]]
            city2 = self.cities[chromosome[(i + 1) % len(chromosome)]]
            total_distance += math.sqrt((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2)
        return 1 / total_distance
    
    def tournament_selection(self, population: List[List[int]]) -> List[int]:
        """Select chromosome using tournament selection."""
        tournament = random.sample(population, self.tournament_size)
        return max(tournament, key=self.calculate_fitness)
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Perform ordered crossover between parents."""
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child1 = [None] * len(parent1)
        child2 = [None] * len(parent2)
        
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        fill_child(child1, parent2, end)
        fill_child(child2, parent1, end)
        
        return child1, child2
    
    def mutate(self, chromosome: List[int]) -> List[int]:
        """Apply inversion to chromosome."""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(chromosome)), 2)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome
    
    def solve(self) -> Tuple[List[int], float]:
        """
        Solve the TSP using genetic algorithm.
        
        Returns:
            Tuple containing best solution and its fitness value
        """
        population = self.create_initial_population()
        best_solution = None
        best_fitness = 0
        
        for generation in range(self.generations):
            new_population = []
            current_best = max(population, key=self.calculate_fitness)
            current_best_fitness = self.calculate_fitness(current_best)
            
            if current_best_fitness > best_fitness:
                best_solution = current_best
                best_fitness = current_best_fitness
            
            self.best_fitness_history.append(best_fitness)
            new_population.append(current_best)
            
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([self.mutate(child1), self.mutate(child2)])
            
            population = new_population[:self.population_size]
            logging.info(f"Generation {generation}: Best fitness = {best_fitness}")
            
        return best_solution, best_fitness

def fill_child(child: List[int], parent: List[int], end: int) -> None:
    """Helper function to fill the child chromosome with remaining genes from the parent."""
    current_pos = end
    for gene in parent:
        if gene not in child:
            if current_pos >= len(child):
                current_pos = 0
            child[current_pos] = gene
            current_pos += 1

if __name__ == "__main__":

    num_cities = 100
    cities = [City(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]
    
    ga = GeneticTSP(cities)
    solution, fitness = ga.solve()
    
    print("Best solution:", solution)
    print("Total distance:", 1 / fitness)
    print("City order:")
    for city_idx in solution:
        print(f"City {city_idx}: {cities[city_idx]}")
