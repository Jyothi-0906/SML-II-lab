
import numpy as np
import random

# Distance matrix (0 means no direct connection)
graph = np.array([
    [0, 2, 4, 0, 0],
    [2, 0, 1, 7, 0],
    [4, 1, 0, 3, 5],
    [0, 7, 3, 0, 2],
    [0, 0, 5, 2, 0]
])

n = len(graph)
start = 0
end = 4

# ACO parameters
num_ants = 10
iterations = 100
alpha = 1       # Pheromone importance
beta = 2        # Distance importance
evaporation = 0.5
Q = 100

# Initialize pheromone and heuristic information
pheromone = np.ones((n, n))
heuristic = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if graph[i][j] > 0:
            heuristic[i][j] = 1 / graph[i][j]

best_path = None
best_distance = float("inf")

for iteration in range(iterations):
    paths = []
    distances = []

    for ant in range(num_ants):
        path = [start]
        visited = {start}
        current = start

        while current != end:
            neighbors = [
                j for j in range(n)
                if graph[current][j] > 0 and j not in visited
            ]

            if not neighbors:
                break

            probabilities = [
                (pheromone[current][j] ** alpha) *
                (heuristic[current][j] ** beta)
                for j in neighbors
            ]

            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]

            next_node = random.choices(
                neighbors, weights=probabilities, k=1
            )[0]

            path.append(next_node)
            visited.add(next_node)
            current = next_node

        if current == end:
            distance = sum(
                graph[path[i]][path[i + 1]]
                for i in range(len(path) - 1)
            )

            paths.append(path)
            distances.append(distance)

            if distance < best_distance:
                best_distance = distance
                best_path = path.copy()

    # Evaporate pheromones
    pheromone *= (1 - evaporation)

    # Deposit pheromones on successful paths
    for path, distance in zip(paths, distances):
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            pheromone[a][b] += Q / distance
            pheromone[b][a] += Q / distance

print("Best path:", [node + 1 for node in best_path]
      if best_path else "No path found")

if best_path:
    print("Shortest distance found:", best_distance)
