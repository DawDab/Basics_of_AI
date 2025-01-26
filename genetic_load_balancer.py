from dataclasses import dataclass
from typing import List, Tuple
import random
import logging

@dataclass
class Task:
    """Represents an task with execution time."""
    id: int
    execution_time: float
    
@dataclass
class Processor:
    """Represents a processor with multiplayer."""
    id: int
    multiplayer: float

class GeneticTaskAllocation:
    """Genetic algorithm implementation for task allocation."""
    
    def __init__(self, tasks: List[Task], processors: List[Processor],
        population_size: int = 1000, generations: int = 5000,
        mutation_rate: float = 0.01, tournament_size: int = 3)-> None:
        """
        Initialize the genetic algorithm solver.
        
        Args:
            tasks: List of tasks to allocation
            processors: List of processors available for allocation
            population_size: Size of the population in each generation
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation for each gene
            tournament_size: Number of chromosomes in tournament selection
        """
        self._validate_inputs(tasks, processors, population_size, generations)
        self.tasks = tasks
        self.processors = processors
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.best_fitness_history = []
        logging.basicConfig(level=logging.INFO)
        
    def _validate_inputs(self, tasks: List[Task], processors: List[Processor],
                        population_size: int, generations: int) -> None:
        """Validate input parameters."""
        if not tasks:
            raise ValueError("Tasks list cannot be empty")
        if any(task.execution_time <= 0 for task in tasks):
            raise ValueError("Execution time must be positive")
        if not processors:
            raise ValueError("Processors list cannot be empty")
        if population_size <= 0 or generations <= 0:
            raise ValueError("Population size and generations must be positive")
        
    def create_initial_population(self) -> List[List[int]]:
        """Create random initial population of chromosomes."""
        return [[random.randint(0, len(self.processors) - 1) for _ in self.tasks] 
                for _ in range(self.population_size)]
             
    def calculate_fitness(self, chromosome: List[int]) -> float:
        """
        Calculate fitness value for a chromosome. 
        Fitness is makespan for given task allocation.
        Lower value is better.
        
        Args:
            chromosome: List of processor assignments for each task
        
        Returns:
            float: Maximum execution time across all processors (makespan)
        """
        processor_times = [0.0] * len(self.processors)

        for task_idx, processor_idx in enumerate(chromosome):
            task_time = self.tasks[task_idx].execution_time
            processor_multiplier = self.processors[processor_idx].multiplayer
            processor_times[processor_idx] += task_time * processor_multiplier

        return max(processor_times)
    
    def tournament_selection(self, population: List[List[int]]) -> List[int]:
        """Select chromosome using tournament selection."""
        tournament = random.sample(population, self.tournament_size)
        return min(tournament, key=self.calculate_fitness)
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Perform single-point crossover between parents."""
        point = random.randint(1, len(self.tasks)-1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def mutate(self, chromosome: List[int]) -> List[int]:
        """Apply mutation to all tasks with probability."""
        return [random.randint(0, len(self.processors) - 1) if random.random() < self.mutation_rate else gene 
                for gene in chromosome] 
        
    def solve(self) -> Tuple[List[int], float]:
        """
        Solve the task allocation problem using genetic algorithm.
        
        Returns:
            Tuple containing best solution and its fitness value
        """
        population = self.create_initial_population()
        best_solution = None
        best_fitness = float('inf')
        
        for generation in range(self.generations):
            new_population = []
            current_best = min(population, key=self.calculate_fitness)
            current_best_fitness = self.calculate_fitness(current_best)
            
            if current_best_fitness < best_fitness:
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
            print(f"Generation {generation}: Best fitness = {best_fitness}")
            
        return best_solution, best_fitness

if __name__ == "__main__":
    # Generate random tasks and processors
    num_tasks = 100
    tasks = [Task(i, random.uniform(10, 90)) for i in range(num_tasks)]

    processors = [
        Processor(0, 1.0),
        Processor(1, 1.25),
        Processor(2, 1.5),
        Processor(3, 1.75)
    ]

    # Solve the task allocation problem
    ga = GeneticTaskAllocation(tasks, processors)
    solution, fitness = ga.solve()
    
    print("Best solution:", solution)
    print("Makespan:", fitness)
    print("Task allocation:")
    for task_idx, processor_idx in enumerate(solution):
        print(f"Task {task_idx} -> Processor {processor_idx}")