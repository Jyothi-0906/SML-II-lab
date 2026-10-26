
import math
import random

# Objective function
def f(x):
    return x**2 + 10 * math.sin(x)

# Simulated Annealing parameters
lower_bound = -10
upper_bound = 10

temperature = 100
cooling_rate = 0.95
min_temperature = 0.001
iterations = 100

# Initialize the current solution
current_x = random.uniform(lower_bound, upper_bound)
current_value = f(current_x)

# Store the best solution
best_x = current_x
best_value = current_value

# Simulated Annealing loop
while temperature > min_temperature:

    for i in range(iterations):

        # Generate a neighboring solution
        new_x = current_x + random.uniform(-1, 1)

        # Keep the solution within the bounds
        new_x = max(lower_bound, min(upper_bound, new_x))

        # Evaluate the new solution
        new_value = f(new_x)

        # Calculate the change in objective value
        delta = new_value - current_value

        # Accept better solutions automatically
        # Accept worse solutions with a probability
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_x = new_x
            current_value = new_value

        # Update the best solution found
        if current_value < best_value:
            best_x = current_x
            best_value = current_value

    # Reduce the temperature
    temperature *= cooling_rate

print("Best x:", best_x)
print("Minimum function value:", best_value)
