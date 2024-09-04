import numpy as np
import random
import copy
import kmedoids
from typing import Any
from tqdm import tqdm

from pan_biclustering.hypervolume import HyperVolume



class PAN:
    """
    Python implementation of the PAN algorithm as defined by Tamara Ulrich.
    Ulrich, T. (2013), Pareto-Set Analysis: Biobjective Clustering in Decision and Objective Spaces. J. Multi-Crit. Decis. Anal., 20: 217-234. https://doi.org/10.1002/mcda.1477
    
    PAN performs biclustering in the decision and objective spaces, finding clusters that are optimal in both spaces.
    These biclusters can be used to analyse pareto fronts in multi-criteria decision making.
    PAN is a genetic multi-objective algorithm, using the basic k-medoids clustering algorithm as a heuristic to optimize the clusters.
    
    This implementation uses a direct representation of partitionings and the silhouettes validity index.
    """
    
    decision_dissimilarity_matrix: np.ndarray[np.ndarray[float]]
    objective_dissimilarity_matrix: np.ndarray[np.ndarray[float]]
    number_of_solutions: int
    hv: HyperVolume


    def __init__(self, decision_space: list[tuple[Any]], objective_space: list[tuple[int]], decision_distance: callable) -> None:
        """
        Create a new PAN instance. 
        Calculates distance matrices for the decision and objective spaces, which are used when finding clusters.
        """
        def euclidean_distance(d1, d2) -> float:
            return sum((d1[i] - d2[i])**2 for i in range(len(d1)))**0.5
        
        assert len(decision_space) == len(objective_space), "The decision and objective spaces must have the same number of solutions"
        
        self.decision_dissimilarity_matrix = self.create_dissimilarity_matrix(decision_space, decision_distance)
        self.objective_dissimilarity_matrix = self.create_dissimilarity_matrix(objective_space, euclidean_distance)
        
        self.number_of_solutions = len(decision_space)
        
        # Reference point set to (1, 1), as the objectives are silhouette indices ranging [-1, 1]
        self.hv = HyperVolume((1, 1))
        
    
    @classmethod
    def from_dissimilarity_matrix(cls, decision_dissimilarity_matrix: np.ndarray[np.ndarray[float]], objective_dissimilarity_matrix: np.ndarray[np.ndarray[float]]):
        """
        Create a new PAN instance from precalculated distance matrices
        """
        pan = cls.__new__(cls)
        pan.decision_dissimilarity_matrix = decision_dissimilarity_matrix
        pan.objective_dissimilarity_matrix = objective_dissimilarity_matrix
        pan.number_of_solutions = len(decision_dissimilarity_matrix)
        pan.hv = HyperVolume((1, 1))
        return pan


    def create_dissimilarity_matrix(self, space: list[tuple[Any]], distance: callable) -> np.ndarray[np.ndarray[float]]:
        """
        Creates a square matrix containing distances between any two points using the given distance function.
        """
        dissimilarity_matrix = []
        for i in range(len(space)):
            row = []
            for j in range(len(space)):
                row.append(distance(space[i], space[j]))
            dissimilarity_matrix.append(row)
        return np.array(dissimilarity_matrix)


    def silhouette_indices(self, labels: list[int]) -> tuple[float, float]:
        """
        Calculates the silhouette indices for the given cluster labels.
        The silhouette index is a measure of how similar an object is to its own cluster compared to other clusters.
        """
        decision_silhouette = kmedoids.silhouette(self.decision_dissimilarity_matrix, labels)[0]
        objective_silhouette = kmedoids.silhouette(self.objective_dissimilarity_matrix, labels)[0]
        return (decision_silhouette, objective_silhouette)       
    
    
    def calculate_population_indices(self, population: list[list[list[int]]]) -> list[tuple[float, float]]:
        """
        Retrieves the silhouette indices for each partitioning in the population.
        """
        population_indices = []
        for partitioning in population:
            # Create an array containing the cluster index for each solution
            labels = [[j for j in range(len(partitioning)) if i in partitioning[j]][0] for i in range(self.number_of_solutions)]
            population_indices.append(self.silhouette_indices(labels))
            
        return population_indices
    
    
    def is_valid(self, partitioning: list[list[int]]) -> bool:
        """
        Checks whether the partitioning is valid, i.e.:
        - Contains all solutions exactly once
        - Contains more than 1 cluster
        - All clusters contain more than 1 solution
        """
        solutions = [solution for cluster in partitioning for solution in cluster]
        return len(solutions) == len(set(solutions)) == self.number_of_solutions\
            and len(partitioning) > 1 \
            and all(len(cluster) > 1 for cluster in partitioning)

    
    def is_invalid(self, partitioning) -> bool:
        """
        Checks whether the given partitioning is invalid
        """
        return not self.is_valid(partitioning)
            
    
    def remove_empty_clusters(self, partitioning: list[list[int]]) -> list[list[int]]:
        """
        Removes empty clusters from a partitioning
        """
        return [cluster for cluster in partitioning if cluster]
    
    
    def get_random_partitioning(self) -> list[list[int]]:
        """
        Generates a random partitioning of the solutions
        """
        num_clusters = random.randint(1, self.number_of_solutions // 2)
        partitioning = [[] for _ in range(num_clusters)]
        
        for i in range(self.number_of_solutions):
            cluster_index = random.randint(0, num_clusters - 1)
            partitioning[cluster_index].append(i)
            
        partitioning = self.remove_empty_clusters(partitioning)
        return partitioning
    
    
    def get_random_valid_partitioning(self) -> list[list[int]]:
        """
        Retrieves a random valid partitioning
        """
        partitioning = self.get_random_partitioning()
        while self.is_invalid(partitioning):
            partitioning = self.get_random_partitioning()
        return partitioning
    
    
    def get_random_invalid_partitioning(self) -> list[list[int]]:
        """
        Retrieves a random invalid partitioning
        """
        partitioning = self.get_random_partitioning()
        while self.is_valid(partitioning):
            partitioning = self.get_random_partitioning()
        return partitioning


    def random_population(self, n) -> list[list[list[int]]]:
        """
        Generate a random population of n partitionings
        """
        return list(self.get_random_valid_partitioning() for _ in range(n))


    def mating_selection(self, population: list[list[list[int]]]) -> tuple[list[list[int]], list[list[int]]]:
        """
        Randomly select 2 partitionings from the population
        """
        partitioning_1, partitioning2 = random.sample(population, 2)
        return copy.deepcopy(partitioning_1), copy.deepcopy(partitioning2)

    
    def recombine(self, parent1: list[list[int]], parent2: list[list[int]], pr: float) -> tuple[list[list[int]], list[list[int]]]:
        """
        With probability pr, recombine the two partitionings
        Recombination is done as defined in the paper, with an operationing resembling a two-point crossover
        """
        # Helper functions
        def cut_points(partitioning):
            # Define cut points for the two-point crossover
            n = len(partitioning)
            i, j = random.sample(range(n + 1), 2)
            if i > j:
                i, j = j, i
            return i, j
        
        def combine(partitioning, partitioning_slice):
            # Place the slice in the partitioning
            # Remove solutions from p that are in the slice
            offspring = []
            for old_cluster in partitioning:
                cluster = []
                for solution in old_cluster:
                    if sum(solution in new_cluster for new_cluster in partitioning_slice) == 0:
                        cluster.append(solution)
                offspring.append(cluster)
            offspring += partitioning_slice
            return offspring
        
        # Perform recombination with probability pr
        if random.random() < pr:
            # Perform recombination
            i1, j1 = cut_points(parent1)
            i2, j2 = cut_points(parent2)
            
            slice1 = parent1[i1:j1]
            slice2 = parent2[i2:j2]
            
            offspring1 = combine(parent1, slice2)
            offspring2 = combine(parent2, slice1)
        else:
            # Don't change the partitionings
            offspring1, offspring2 = parent1, parent2
        offspring1 = self.remove_empty_clusters(offspring1)
        offspring2 = self.remove_empty_clusters(offspring2)
        return offspring1, offspring2
    
    
    def move(self, partitioning: list[list[int]]) -> list[list[int]]:
        """
        Moves a randomly selected solution to a randomly selected other cluster
        """
        if len(partitioning) < 2:
            # A partitioning with only 1 cluster can't move solutions
            return partitioning
        
        cluster_from, cluster_to = random.sample(range(len(partitioning)), 2)
        if not partitioning[cluster_from]:
            # A cluster with no solutions can't move solutions
            return partitioning
        
        solution = random.choice(partitioning[cluster_from])
        
        del partitioning[cluster_from][partitioning[cluster_from].index(solution)]
        partitioning[cluster_to].append(solution)
        
        partitioning = self.remove_empty_clusters(partitioning)
        return partitioning
        
    
    def merge(self, partitioning: list[list[int]]) -> list[list[int]]:
        """
        Randomly select two clusters and merge them
        """
        if len(partitioning) < 2:
            # A partitioning with only 1 cluster can't merge clusters
            return partitioning
        
        cluster_1, cluster_2 = random.sample(range(len(partitioning)), 2)
        new_cluster = partitioning[cluster_1] + partitioning[cluster_2]
        
        # Ensure that the cluster with the highest index is removed first
        if cluster_2 > cluster_1:
            cluster_1, cluster_2 = cluster_2, cluster_1
                
        del partitioning[cluster_1]
        del partitioning[cluster_2]
        
        partitioning.append(new_cluster)
        
        partitioning = self.remove_empty_clusters(partitioning)
        return partitioning        
    
    
    def split(self, partitioning: list[list[int]]) -> list[list[int]]:
        """
        Split a randomly selected cluster into two random parts
        """
        if len(partitioning) < 1:
            # A partitioning with no clusters can't be split            
            return partitioning
        
        cluster = random.choice(range(len(partitioning)))
        
        if len(partitioning[cluster]) == 1:
            # A cluster with only 1 solution can't be split
            return partitioning
        
        split = random.randint(1, len(partitioning[cluster]))
        
        cluster_1 = partitioning[cluster][:split]
        cluster_2 = partitioning[cluster][split:]
        
        del partitioning[cluster]
        partitioning.append(cluster_1)
        partitioning.append(cluster_2)
        
        partitioning = self.remove_empty_clusters(partitioning)
        return partitioning
    
    
    def mutate(self, partitioning: list[list[int]], pm: float, pu: float, ps: float) -> list[list[int]]:
        """
        Perform mutations on the given partitioning with the given probabilities.
        """
        if random.random() < pm:
            partitioning = self.move(partitioning)
        if random.random() < pu:
            partitioning = self.merge(partitioning)
        if random.random() < ps:
            partitioning = self.split(partitioning)
        return partitioning
            

    def remove_duplicates(self, population: list[list[list[int]]]) -> list[list[list[int]]]:
        """
        Removes duplicate partitionings from the list of partitionings
        """
        # Sort the clusters and partitionings such that they are comparable
        population = [sorted([sorted(c) for c in p]) for p in population]
        
        # Find duplicate partitionings
        unique_population = []
        for partitioning in population:
            if partitioning not in unique_population:
                unique_population.append(partitioning)
        return unique_population
    
    
    def hyp(self, validity_indices: list[tuple[float, float]]) -> float:
        """
        Calculate the hypervolume of the given validity indices
        """
        return self.hv.compute(validity_indices)


    def select(self, population: list[list[list[int]]], n: int) -> list[list[list[int]]]:
        """
        Select the n partitionings from the given population that contribute most to the total hypervolume
        """
        population = self.remove_duplicates(population)
        
        validity_indices = list(enumerate(self.calculate_population_indices(population)))
        
        while len(validity_indices) > n:
            def key(i):
                copied_indices = validity_indices.copy()
                # Find the index of this solution in the copied list
                for j in range(len(copied_indices)):
                    if copied_indices[j][0] == i:
                        break
                del copied_indices[j]
                return self.hyp([s for _, s in validity_indices]) - self.hyp([s for _, s in copied_indices])
            
            validity_indices.remove(min(validity_indices, key=lambda x: key(x[0])))
        
        population = [population[i] for i, _ in validity_indices]
        
        return population    


    def variate(self, population: list[list[list[int]]], n: int, pr: float, pm: float, pu: float, ps: float) -> list[list[list[int]]]:
        """
        Create a new population by recombining and mutating the given population
        """
        varied_population = []
        for _ in range(1, n//2):
            offspring1 = self.get_random_invalid_partitioning()
            offspring2 = self.get_random_invalid_partitioning()
            
            # Until both offspring are valid
            while self.is_invalid(offspring1) or self.is_invalid(offspring2):
                parent1, parent2 = self.mating_selection(population)
                parent1, parent2 = self.recombine(parent1, parent2, pr)
                
                parent1 = self.mutate(parent1, pm, pu, ps)
                parent2 = self.mutate(parent2, pm, pu, ps)
                
                if self.is_invalid(offspring1) and self.is_valid(parent1):
                    offspring1 = parent1
                if self.is_invalid(offspring2) and self.is_valid(parent2):
                    offspring2 = parent2

            offspring1 = self.remove_empty_clusters(offspring1)
            offspring2 = self.remove_empty_clusters(offspring2)
            varied_population += [offspring1, offspring2]
        return varied_population
    

    def medoid_clustering(self, partitioning: list[list[int]], dissimilarity_matrix: np.ndarray[np.ndarray[float]]) -> list[list[int]]:
        """
        Perform local optimization on a partitioning with a single objective (given by the distance matrix) using k-medoids clustering
        """
        # Calculate the medoids
        medoids = []
        for cluster in partitioning:
            medoids.append(np.argmin(dissimilarity_matrix[cluster].sum(axis=0)))
        medoids = np.array(medoids)
        
        # Let k-medoids cluster the solutions
        labels = kmedoids.fasterpam(dissimilarity_matrix, medoids).labels
        
        # Transform labels list to a partitioning representation list
        partitioning = [[] for _ in range(len(partitioning))]
        for i in range(len(labels)):
            partitioning[labels[i]].append(i)
        return partitioning


    def local_optimization(self, population: list[list[list[int]]]) -> list[list[list[int]]]:
        """
        Localy optimize all partitionings in the population using k-medoids clustering for both ojectives
        """
        optimized_population = []
        for partitioning in population:
            optimized_population.append(self.medoid_clustering(partitioning, self.decision_dissimilarity_matrix))
            optimized_population.append(self.medoid_clustering(partitioning, self.objective_dissimilarity_matrix))
        return optimized_population
            
            
    def sort_population(self, population: list[list[list[int]]], population_indices: list[tuple[float, float]]) -> tuple[list[list[list[int]]], list[tuple[float, float]]]:
        """
        Sort the population based on their population indices
        Ensures that the first returned partitioning is best in clustering the decision space, and the last best in the objective space
        """
        paired_lists = list(zip(population, population_indices))
        # Sort pairs based on the indices
        sorted_pairs = sorted(paired_lists, key=lambda x: x[1])
        sorted_population, sorted_indices = zip(*sorted_pairs)
        return list(sorted_population), list(sorted_indices)


    def find_clusters(self, n: int = 10, g: int = 500, pr: float = 0.7, pm: float = 0.6, pu: float = 0.2, ps: float = 0.2, use_heuristic: bool =True) -> tuple[list[list[list[int]]], list[tuple[float, float]]]:
        """
        Perform the PAN algorithm to find clusters in the decision and objective spaces
        n - number of partitionings to generate
        g - number of generations to run the algorithm
        pr - probability of recombination
        pm - probability of mutation
        pu - probability of merging
        ps - probability of splitting
        use_heuristic - whether to use local optimization using k-medoids clustering
        
        Returns a tuple containing the population and their indices
        Population is a list of partitionings, where each partitioning is a list of clusters, where each cluster is a list of solution indices.
        Population indices is a list of tuples, where each tuple contains the silhouette indices for the decision and objective spaces, indicating the quality of the partitioning in each space.
        """
        population = self.random_population(n)
        for _ in tqdm(range(g)):
            varied_population = self.variate(population, n, pr, pm, pu, ps)
            locally_optimized_population = self.local_optimization(varied_population) if use_heuristic else []
            population = self.select(population + varied_population + locally_optimized_population, n)
        population_indices = self.calculate_population_indices(population)
        population, population_indices = self.sort_population(population, population_indices)
        return population, population_indices
