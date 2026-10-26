
import numpy as np

# Rosenbrock function
def rosenbrock(position):
    x = position[0]
    y = position[1]
    return (1 - x)**2 + 100 * (y - x**2)**2


# PSO parameters
num_particles = 30
dimensions = 2
iterations = 1000

w = 0.7       # Inertia weight
c1 = 1.5      # Cognitive coefficient
c2 = 1.5      # Social coefficient

lower_bound = -5
upper_bound = 5

# Initialize particle positions and velocities
positions = np.random.uniform(
    lower_bound, upper_bound,
    (num_particles, dimensions)
)

velocities = np.random.uniform(
    -1, 1,
    (num_particles, dimensions)
)

# Initialize personal best positions
personal_best_positions = positions.copy()
personal_best_values = np.array([
    rosenbrock(p) for p in positions
])

# Initialize global best position
best_index = np.argmin(personal_best_values)
global_best_position = personal_best_positions[best_index].copy()
global_best_value = personal_best_values[best_index]

# PSO optimization loop
for iteration in range(iterations):

    for i in range(num_particles):

        # Evaluate current particle
        fitness = rosenbrock(positions[i])

        # Update personal best
        if fitness < personal_best_values[i]:
            personal_best_values[i] = fitness
            personal_best_positions[i] = positions[i].copy()

        # Update global best
        if fitness < global_best_value:
            global_best_value = fitness
            global_best_position = positions[i].copy()

        # Generate random numbers
        r1 = np.random.random(dimensions)
        r2 = np.random.random(dimensions)

        # Update velocity
        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (
                personal_best_positions[i] - positions[i]
            )
            + c2 * r2 * (
                global_best_position - positions[i]
            )
        )

        # Update position
        positions[i] = positions[i] + velocities[i]

        # Keep particles inside the search bounds
        positions[i] = np.clip(
            positions[i], lower_bound, upper_bound
        )

    # Display progress every 100 iterations
    if (iteration + 1) % 100 == 0:
        print(
            f"Iteration {iteration + 1}: "
            f"Best value = {global_best_value:.8f}"
        )

# Final result
print("\nOptimization completed")
print("Best x =", global_best_position[0])
print("Best y =", global_best_position[1])
print("Minimum function value =", global_best_value)
