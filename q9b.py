import pandas as pd
import numpy as np
 
MIN_X, MAX_X, L = 0, 2, 5
def f(x):
    return -x**2 + 2*x
 
def decode(chromosome: str) -> float:
    decimal_value = int(chromosome, 2)
    return MIN_X + (MAX_X - MIN_X) / (2**L - 1) * decimal_value
 
# ------------------------------------------------------------
# Step 1: Initial population, decoded x, f(x)
# ------------------------------------------------------------
population = ['11010', '00111', '10110', '00101']
x_values = [decode(c) for c in population]
f_values = [f(x) for x in x_values]
 
# ------------------------------------------------------------
# Step 2: Minimization fitness transform
# ------------------------------------------------------------
fitness = [1 / (1 + fv) for fv in f_values]
total_fitness = sum(fitness)
prob = [fit / total_fitness for fit in fitness]
cumulative_prob = np.cumsum(prob)
 
# ------------------------------------------------------------
# Step 3: Selection using the given random numbers
# ------------------------------------------------------------
random_numbers = [0.40, 0.15, 0.70, 0.90]
 
def select(r, cum_prob):
    for i, cp in enumerate(cum_prob):
        if r <= cp:
            return i
    return len(cum_prob) - 1
 
selected_indices = [select(r, cumulative_prob) for r in random_numbers]
mating_pool = [population[i] for i in selected_indices]
 
# ------------------------------------------------------------
# Step 4: Crossover (single-point, site = 1)
# ------------------------------------------------------------
def crossover(parent_a, parent_b, site):
    child1 = parent_a[:site] + parent_b[site:]
    child2 = parent_b[:site] + parent_a[site:]
    return child1, child2
 
SITE = 1
pair1 = (mating_pool[0], mating_pool[1])
pair2 = (mating_pool[2], mating_pool[3])
 
child1, child2 = crossover(pair1[0], pair1[1], SITE)
child3, child4 = crossover(pair2[0], pair2[1], SITE)
new_population = [child1, child2, child3, child4]
 
# ------------------------------------------------------------
# Step 5: New population - decode and evaluate
# ------------------------------------------------------------
x_values_2 = [decode(c) for c in new_population]
f_values_2 = [f(x) for x in x_values_2]
 
# ------------------------------------------------------------
# Step 6: Best solution across both generations
# ------------------------------------------------------------
all_x = x_values + x_values_2
all_f = f_values + f_values_2
all_chromo = population + new_population
best_idx = int(np.argmin(all_f))          # minimization -> smallest f(x) wins
 
print("Generation 1:", list(zip(population, np.round(x_values,4), np.round(f_values,4))))
print("Mating pool:", mating_pool)
print("Generation 2:", list(zip(new_population, np.round(x_values_2,4), np.round(f_values_2,4))))
print(f"Best solution found: chromosome={all_chromo[best_idx]}, "
      f"x={all_x[best_idx]:.4f}, f(x)={all_f[best_idx]:.4f}")
 
# ------------------------------------------------------------
# Analytical verification
# ------------------------------------------------------------
import sympy as sp
x = sp.symbols('x')
f_sym = -x**2 + 2*x
critical = sp.solve(sp.Eq(sp.diff(f_sym, x), 0), x)
print(f"f(0)={f_sym.subs(x,0)}, f(2)={f_sym.subs(x,2)}  -> true minimum is 0, at x=0 and x=2")
