import pandas as pd
import numpy as np
 
def f(x):
    return x**2
 
def decode(chromosome: str) -> int:
    return int(chromosome, 2)
 
# ------------------------------------------------------------
# Step 1: Initial population and fitness
# ------------------------------------------------------------
population = ['01100', '11001', '00101', '10011']
x_values = [decode(c) for c in population]
fitness = [f(x) for x in x_values]
 
# ------------------------------------------------------------
# Step 2: Selection probability, expected count, actual count
# ------------------------------------------------------------
total_fitness = sum(fitness)
avg_fitness = total_fitness / len(fitness)
 
prob = [fit / total_fitness for fit in fitness]
expected_count = [fit / avg_fitness for fit in fitness]
actual_count = [1, 2, 0, 1]        # rounded roulette-wheel counts (sum = N = 4)
 
# ------------------------------------------------------------
# Step 3: Mating pool
# ------------------------------------------------------------
mating_pool = []
for chromo, count in zip(population, actual_count):
    mating_pool.extend([chromo] * count)
 
# ------------------------------------------------------------
# Step 4: Crossover (single-point)
# ------------------------------------------------------------
def crossover(parent_a, parent_b, site):
    child1 = parent_a[:site] + parent_b[site:]
    child2 = parent_b[:site] + parent_a[site:]
    return child1, child2
 
pair1 = (mating_pool[0], mating_pool[1])
pair2 = (mating_pool[2], mating_pool[3])
 
child1, child2 = crossover(pair1[0], pair1[1], site=4)
child3, child4 = crossover(pair2[0], pair2[1], site=2)
 
new_population = [child1, child2, child3, child4]
 
# ------------------------------------------------------------
# Step 5: Mutation operator (bit-flip) - demonstration
# ------------------------------------------------------------
def mutate_bit(chromosome, position):
    bits = list(chromosome)
    bits[position] = '1' if bits[position] == '0' else '0'
    return ''.join(bits)
 
# ------------------------------------------------------------
# Step 6: New population fitness (Generation 2)
# ------------------------------------------------------------
x_values_2 = [decode(c) for c in new_population]
fitness_2 = [f(x) for x in x_values_2]
 
print("Generation 1:", list(zip(population, x_values, fitness)))
print("Mating pool:", mating_pool)
print("Generation 2:", list(zip(new_population, x_values_2, fitness_2)))
print(f"Gen1 -> Max f(x)={max(fitness)}, Avg f(x)={avg_fitness:.2f}")
print(f"Gen2 -> Max f(x)={max(fitness_2)}, Avg f(x)={np.mean(fitness_2):.2f}")
