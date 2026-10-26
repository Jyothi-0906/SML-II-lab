
import random
import math

# Coordinates of five cities
cities = {
    0: (0, 0),
    1: (2, 6),
    2: (5, 3),
    3: (6, 7),
    4: (8, 2)
}

num_cities = len(cities)

# Calculate Euclidean distance between two cities
def distance(city1, city2):
    x1, y1 = cities[city1]
    x2, y2 = cities[city2]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Calculate total distance of a tour
def tour_distance(tour):
    total = 0

    for i in range(len(tour)):
        city1 = tour[i]
        city2 = tour[(i + 1) % len(tour)]
        total += distance(city1, city2)

    return total

# Genetic Algorithm parameters
population_size = 50
generations = 200
mutation_rate = 0.1
tournament_size = 3

# Create a random tour
def create_tour():
    return random.sample(range(num_cities), num_cities)

# Initialize population
population = [
    create_tour() for _ in range(population_size)
]

# Selection using tournament selection
def selection(population):
    candidates = random.sample(
        population, tournament_size
    )
    return min(candidates, key=tour_distance)

# Ordered crossover
def crossover(parent1, parent2):
    start, end = sorted(
        random.sample(range(num_cities), 2)
    )

    child = [None] * num_cities
    child[start:end + 1] = parent1[start:end + 1]

    remaining = [
        city for city in parent2 if city not in child
    ]

    j = 0
    for i in range(num_cities):
        if child[i] is None:
            child[i] = remaining[j]
            j += 1

    return child

# Swap mutation
def mutate(tour):
    if random.random() < mutation_rate:
        i, j = random.sample(range(num_cities), 2)
        tour[i], tour[j] = tour[j], tour[i]

    return tour

# Find the best tour
best_tour = min(population, key=tour_distance)
best_distance = tour_distance(best_tour)

# Main Genetic Algorithm loop
for generation in range(generations):

    new_population = []

    # Preserve the best tour (elitism)
    new_population.append(best_tour.copy())

    while len(new_population) < population_size:

        # Select parents
        parent1 = selection(population)
        parent2 = selection(population)

        # Create a child
        child = crossover(parent1, parent2)

        # Mutate the child
        child = mutate(child)

        new_population.append(child)

    population = new_population

    # Update the best tour
    current_best = min(population, key=tour_distance)
    current_distance = tour_distance(current_best)

    if current_distance < best_distance:
        best_tour = current_best.copy()
        best_distance = current_distance

    if (generation + 1) % 20 == 0:
        print(
            f"Generation {generation + 1}: "
            f"Best distance = {best_distance:.2f}"
        )

# Display final result
print("\nBest tour:", best_tour + [best_tour[0]])
print("Shortest distance found:", best_distance)
