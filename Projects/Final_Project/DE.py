import numpy as np
import random
import torch
from mpi4py import MPI


# Fitness Function: Quadratic Error
def quadratic_error(label, pred):
    sum_ = 0
    for _ in range(len(pred)):
        sum_ += np.sum((label - pred) ** 2)

    return sum_
    

# Function that contains differential evolution algorithm
def differential_evolution(labels, predictions, dim, pop_size=20, func=quadratic_error, li=-10, ls=10, gen=30):

    if pop_size < 4: raise Exception('Population size must be >= 4')
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Initialize population of D-dimensions within the range
        population = np.random.uniform(li, ls, (pop_size, dim))
        
    else:
        population = None

    # Broadcast the population to all processes
    population = comm.bcast(population, root=0)

    # Initialize weight and crossover probability
    F = np.random.uniform(0, 2)
    CR = np.random.random()

    #labels = torch.Tensor.numpy(labels)
    labels = np.array([label.detach().numpy() for label in labels])
    predictions = np.array([pred.detach().numpy() for pred in predictions])

    for _ in range(gen):

        # Evaluate individuls with the objective function and store the fitness
        fitness = np.array([func(label, pred) for label, pred in zip(labels, predictions)], dtype=float)
        
        # Array to store the trial individuals after operations
        trial_indivs = np.zeros((pop_size, dim), dtype=float)
        
        # Loop for mutation and crossover operations
        for i in range(pop_size):

            target_v = population[i] # Target vector

            # Choose 2 random individuals from population
            indiv_mut = population[random.sample(range(pop_size), 3)]

            # Determinate donor vector (mutation operation)
            donor_v = indiv_mut[0] + F * (indiv_mut[1] - indiv_mut[2])

            # Generate a random integer, which ensures that the trial vector will contain 
            # at least one individual from the donor vector
            rnd_idx = np.random.randint(1, dim)

            # Generate dim random numbers to compare with CR factor
            r_v = np.random.rand(dim)

            # Array to store the trial vector
            trial_v = np.zeros(dim) 

            # Determinate trial vector (crossover operation)
            for j in range(dim):
                if r_v[j] <= CR or j == rnd_idx:
                    trial_v[j] = donor_v[j]
                elif r_v[j] > CR and j != rnd_idx:
                    trial_v[j] = target_v[j]

            # If any value of the vector is out-bounds the search space, 
            # it'll be set to the corresponding limit
            if np.any(trial_v < li) or np.any(trial_v > ls):
                trial_v[trial_v < li] = li
                trial_v[trial_v > ls] = ls

            # Add the trial vector to the trial_individuals        
            trial_indivs[i] = trial_v
            
        # Calculate trial_individuals fitnesses
        fitness_ti = np.array([func(label, pred) for label, pred in zip(labels, predictions)], dtype=float)

        # Generate new population from fitnesess of population and trial_individuals
        population = np.array([population[i] if fitness[i] < fitness_ti[i] else trial_indivs[i] for i in range(pop_size)], dtype=float)

        # Synchronize the population across all processes
        comm.Bcast(population, root=0)
        comm.Bcast(fitness, root=0)

    # Find the best individual 
    best_idx = np.argmin(fitness)
    optimum = comm.bcast(population[best_idx], root=0)

    return optimum
