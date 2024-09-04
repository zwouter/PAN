import matplotlib.pyplot as plt

from pan_biclustering.pan import PAN


# Points clustered in the decision space
pareto_set = [
    [1, 2], [2, 1], [1, 1], [2, 2], [1.5, 1.5],  # Cluster 1
    [5, 6], [6, 5], [5, 5], [6, 6], [5.5, 5.5],  # Cluster 2
    [9, 10], [10, 9], [9, 9], [10, 10], [9.5, 9.5],  # Cluster 3
    [13, 14], [14, 13], [13, 13], [14, 14], [13.5, 13.5]  # Cluster 4
]

# Points clustered in the objective space
pareto_front = [
    [1, 2], [2, 1], [5, 6], [6, 5], [9, 10],  # Cluster 1
    [1, 1], [2, 2], [5, 5], [6, 6], [9, 9],  # Cluster 2
    [1.5, 1.5], [5.5, 5.5], [9.5, 9.5], [13, 14], [14, 13],  # Cluster 3
    [13, 13], [14, 14], [13.5, 13.5], [10, 10], [10, 9]  # Cluster 4
]

# Normalize the space to [0, 1]
def normalize(space: list[list]):
    normalized_space = []
    for i in range(len(space[0])):
        column = [d[i] for d in space]
        min_val = min(column)
        max_val = max(column)
        normalized_column = [(val - min_val) / (max_val - min_val) for val in column]
        normalized_space.append(normalized_column)
    return list(zip(*normalized_space))

# Normalize the spaces to [0, 1] 
pareto_set = normalize(pareto_set)
pareto_front = normalize(pareto_front)


# Define a distance measure for the decision space
# We can use euclidean distance, as is used in the objective space
def euclidean_distance(p1, p2):
    return sum([(a - b) ** 2 for a, b in zip(p1, p2)]) ** 0.5

# Perform the clustering
number_of_solutions = 5
pan = PAN(pareto_set, pareto_front, euclidean_distance)
population, population_indices = pan.find_clusters(number_of_solutions, 400)


# Plot the different clusters and their validity indices

fig, axs = plt.subplots(2, number_of_solutions + 1)

x, y = zip(*population_indices)
axs[0, 0].scatter(x, y)
axs[1, 0].plot(x, y)

for k in range(2, number_of_solutions + 2):
    partitioning = population[k - 2]
    
    x, y = zip(*pareto_set)
    colors = []
    for i in range(len(x)):
        colors.append([j for j in range(len(partitioning)) if i in partitioning[j]][0])
    axs[0, k - 1].scatter(x, y, c=colors)
    
    x, y = zip(*pareto_front)
    colors = []
    for i in range(len(x)):
        colors.append([j for j in range(len(partitioning)) if i in partitioning[j]][0])
    axs[1, k - 1].scatter(x, y, c=colors)    
    
plt.show()