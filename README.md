# PAN
Python implementation of the PAN biclustering algorithm [[1]](#1).

PAN is made to perform analysis on the Pareto set and front resulting from a multi-objective optimization method. In its turn, PAN is a bi-objective evolutionary algorithm, performing clustering in two spaces at the same time. Since this is another conflicting optimization problem, PAN returns multiple partitionings (collections of clusters), from which the user can choose their preferred partitioning.

PAN makes a few assumptions about the given data:
- The best partitioning in the objective space differs from the best partitioning in the decision space. If this is not the case, a single objective clustering algorithm can be used on one of the two spaces.
- Each cluster contains at least two solutions. If this is not the case, outliers need to be manually removed.

To speedup optimization, PAN uses k-medoids clustering as a local heuristic, implemented here using the fast [kmedoids](https://github.com/kno10/python-kmedoids) package.


## Installation
A pre-built package is available at [Pypi]() and can be installed with:
```sh
pip install pan_biclustering
```


## Usage
A quick example usage can be found below.

```python
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

# Two clusters, both containing two of the points 1, 2, 3 and 4
print(partitioning)
```

A second, more elaborate example can be found in the /examples folder.

To automatically pick a partitioning out of the found partitionings, a knee-point detection algorithm such as [kneed](https://pypi.org/project/kneed/) can be used on the returned population_indices.

## References
<a id="1">1</a>
Ulrich, T. (2013), Pareto-Set Analysis: Biobjective Clustering in Decision and Objective Spaces. J. Multi-Crit. Decis. Anal., 20: 217-234. https://doi.org/10.1002/mcda.1477