from dataclasses import dataclass
from typing import List, Tuple
import random
import logging

@dataclass
class Item:
    """Represents an item in the knapsack problem."""
    weight: float
    value: float

class GeneticKnapsack:
    """Genetic algorithm implementation for solving the knapsack problem."""
    
    def __init__(self, items: List[Item], capacity: float, 
                 population_size: int = 100, generations: int = 1000,
                 mutation_rate: float = 0.1, tournament_size: int = 2) -> None:
        """
        Initialize the genetic algorithm solver.
        
        Args:
            items: List of items available for selection
            capacity: Maximum weight capacity of knapsack
            population_size: Size of the population in each generation
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation for each gene
            tournament_size: Number of chromosomes in tournament selection
        """
        self._validate_inputs(items, capacity, population_size, generations)
        self.items = items
        self.capacity = capacity
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.best_fitness_history = []
        logging.basicConfig(level=logging.INFO)
        
    def _validate_inputs(self, items: List[Item], capacity: float,
                        population_size: int, generations: int) -> None:
        """Validate input parameters."""
        if not items:
            raise ValueError("Items list cannot be empty")
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if population_size <= 0 or generations <= 0:
            raise ValueError("Population size and generations must be positive")
            
    def create_initial_population(self) -> List[List[int]]:
        """Create random initial population of binary chromosomes."""
        return [[random.randint(0, 1) for _ in self.items] 
                for _ in range(self.population_size)]
    
    def calculate_fitness(self, chromosome: List[int]) -> float:
        """Calculate fitness value for a chromosome."""
        total_weight = sum(c * item.weight for c, item in zip(chromosome, self.items))
        if total_weight > self.capacity:
            return 0.0
        return sum(c * item.value for c, item in zip(chromosome, self.items))
    
    def tournament_selection(self, population: List[List[int]]) -> List[int]:
        """Select chromosome using tournament selection."""
        tournament = random.sample(population, self.tournament_size)
        return max(tournament, key=self.calculate_fitness)
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Perform single-point crossover between parents."""
        point = random.randint(1, len(self.items)-1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def mutate(self, chromosome: List[int]) -> List[int]:
        """Apply mutation to chromosome."""
        return [1 - gene if random.random() < self.mutation_rate else gene 
                for gene in chromosome]
    
    def solve(self) -> Tuple[List[int], float]:
        """
        Solve the knapsack problem using genetic algorithm.
        
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

if __name__ == "__main__":
    
    items_data = [(random.uniform(10, 90), random.uniform(10, 90)) for _ in range(100)]
    items = [Item(w, v) for w, v in items_data]
    capacity = 2500
    
    ga = GeneticKnapsack(items, capacity)
    solution, fitness = ga.solve()
    
    print("Best solution:", solution)
    print("Total value:", fitness)
    print("Selected items:")
    for i, selected in enumerate(solution):
        if selected:
            print(f"Item {i}: Weight={items[i].weight}, Value={items[i].value}")