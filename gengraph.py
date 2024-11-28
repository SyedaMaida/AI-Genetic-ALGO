import random
from random import choices, randint, randrange
from typing import List, Callable, Tuple
from functools import partial
from collections import namedtuple
import matplotlib.pyplot as plt

# Type aliases
GENOME = List[int]
POPULATION = List[GENOME]
POPULATEFUNC = Callable[[], POPULATION]
FitnessFunc = Callable[[GENOME], int]
SELECTION = Callable[[POPULATION, FitnessFunc], Tuple[GENOME, GENOME]]
CROSSOVER = Callable[[GENOME, GENOME], Tuple[GENOME, GENOME]]
MUTATION = Callable[[GENOME], GENOME]

# Define the things
THING = namedtuple('Thing', ["name", "value", "weight"])
things = [
    THING("Laptop", 500, 2000),
    THING("HeadPhone", 150, 160),
    THING("Coffee", 60, 350),
    THING("NotePad", 40, 333),
    THING("Water Bottle", 30, 192)
]


# Genetic Algorithm components
def generate_genome(length: int) -> GENOME:
    return choices([0, 1], k=length)


def generate_population(size: int, genome_length: int) -> POPULATION:
    return [generate_genome(genome_length) for _ in range(size)]


def fitness(genome: GENOME, things: List[THING], weight_limit: int) -> int:
    if len(genome) != len(things):
        raise ValueError("The genome and things length must be the same")
    weight = 0
    value = 0
    for i, thing in enumerate(things):
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value
            if weight > weight_limit:
                return 0
    return value


def select_pair(population: POPULATION, fitness_func: FitnessFunc) -> Tuple[GENOME, GENOME]:
    return choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=2
    )


def crossover(a: GENOME, b: GENOME) -> Tuple[GENOME, GENOME]:
    if len(a) != len(b):
        raise ValueError("The lengths of A and B must be the same")
    length = len(a)
    if length < 2:
        return a, b
    p = randint(1, length - 1)
    return a[:p] + b[p:], b[:p] + a[p:]


def mutation(genome: GENOME, num: int = 1, probability: float = 0.01) -> GENOME:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random.random() > probability else abs(genome[index] - 1)
    return genome


# Run evolution with fitness tracking
def run_evolution(
        populatefunc: POPULATEFUNC,
        fitnessfunc: FitnessFunc,
        fitness_limit: int,
        selection_func: SELECTION,
        crossfunc: CROSSOVER,
        mutationfunc: MUTATION,
        generation_limit: int = 100
):
    population = populatefunc()
    best_fitness_over_time = []  # List to track the best fitness

    for i in range(generation_limit):
        # Sort by fitness
        population = sorted(
            population,
            key=lambda genome: fitnessfunc(genome),
            reverse=True
        )

        # Track the best fitness
        best_fitness = fitnessfunc(population[0])
        best_fitness_over_time.append(best_fitness)

        # Check if fitness goal is met
        if best_fitness >= fitness_limit:
            break

        # Generate next generation
        next_generation = population[:2]  # Elitism: keep top 2
        for _ in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitnessfunc)
            offspring1, offspring2 = crossfunc(parents[0], parents[1])
            offspring1 = mutationfunc(offspring1)
            offspring2 = mutationfunc(offspring2)
            next_generation += [offspring1, offspring2]

        population = next_generation

    # Final sort and return
    population = sorted(population, key=lambda genome: fitnessfunc(genome), reverse=True)
    return population[0], i, best_fitness_over_time  # Return fitness data


# Function to map genome to items
def genome_to_things(genome: GENOME, things: List[THING]) -> List[str]:
    result = []
    for i, thing in enumerate(things):
        if genome[i] == 1:
            result.append(thing.name)
    return result


# Execute the genetic algorithm
best_genome, generations, fitness_over_time = run_evolution(
    populatefunc=partial(
        generate_population, size=10, genome_length=len(things)
    ),
    fitnessfunc=partial(
        fitness, things=things, weight_limit=3000
    ),
    fitness_limit=740,
    selection_func=select_pair,
    crossfunc=crossover,
    mutationfunc=mutation,
    generation_limit=100
)

# Plot the fitness graph
plt.plot(range(len(fitness_over_time)), fitness_over_time)
plt.title("Genetic Algorithm Performance")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.show()

# Print results
print(f"No. of Generations: {generations}")
print(f"Best Solution: {genome_to_things(best_genome, things)}")
