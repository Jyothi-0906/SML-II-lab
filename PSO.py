
import numpy as np

# Objective function
def F(x):
    return -x**2 + 5*x + 20

# PSO parameters
C1 = 1
C2 = 1
U1 = 0.213
U2 = 0.876

# Initial particle positions
X = np.array([-9.6, -6, -2.6, -1.1, 0.6,
              2.3, 2.8, 8.3, 10.0])

# Initial velocities
V = np.zeros(len(X))

# Initial personal best positions
Pbest = X.copy()

# Initial global best position (maximization)
Gbest = X[np.argmax(F(X))]

# Number of iterations
iterations = 20

print("Initial positions:", X)
print("Initial fitness:", F(X))
print("Initial global best:", Gbest)

# PSO algorithm
for t in range(iterations):

    # Update velocity
    V = (V
         + C1 * U1 * (Pbest - X)
         + C2 * U2 * (Gbest - X))

    # Update particle positions
    X = X + V

    # Calculate fitness
    fitness = F(X)
    pbest_fitness = F(Pbest)

    # Update personal best for maximization
    better = fitness > pbest_fitness
    Pbest[better] = X[better]

    # Update global best
    best_index = np.argmax(F(Pbest))
    Gbest = Pbest[best_index]

    print(f"Iteration {t+1}:")
    print("Positions:", np.round(X, 4))
    print("Fitness:", np.round(fitness, 4))
    print("Personal best:", np.round(Pbest, 4))
    print("Global best position:", round(Gbest, 4))
    print("Global best fitness:", round(F(Gbest), 4))
    print("-" * 50)

print("\nFinal Result")
print("Best position:", Gbest)
print("Maximum fitness:", F(Gbest))
