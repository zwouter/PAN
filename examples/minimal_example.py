from pan_biclustering.pan import PAN


# Decision space of points 1, 2, 3 and 4
pareto_set = [
    [1, 2], [2, 1], [14, 14], [13, 13],
]
# Objective space of points 1, 2, 3 and 4
pareto_front = [
    [14, 14], [13, 13], [1, 2], [2, 1],
]

def euclidean_distance(p1, p2):
    return sum([(a - b) ** 2 for a, b in zip(p1, p2)]) ** 0.5

pan = PAN(pareto_set, pareto_front, euclidean_distance)
population, population_indices = pan.find_clusters(5, 400)
partitioning = population[0]

# Two clusters, both containing 2 of 1, 2, 3, 4
print(partitioning)