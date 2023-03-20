# Ilse Adriana Pacheco Torres

import numpy as np
import random

# Sphere function
def sphere(indidual):
    return np.sum(np.square(indidual))

# Function to calculate new F parameter
def calculate_F(Fi):
    return 0.1 + np.random.random() * 0.9 if np.random.random() < 0.1 else Fi

# Function to calculate new CR parameter
def calculate_CR(CRi):
    return np.random.random() if np.random.random() < 0.1 else CRi

# Function containing the SaDE algorithm
def sade(pop_size, dim, func, li, ls, gen):

    if pop_size < 4:
        print('For the strategy rand/1/bin, population size must be >= 4')
        exit()

    # Initialize population of D-dimensions within the range
    population = np.random.uniform(li, ls, (pop_size, dim))

    for _ in range(gen):

        # Evaluate individuals with the objective function and store the fitness
        fitness = np.array([func(indiv) for indiv in population])
        
        # Array to store the resulting individuals after the operations
        trial_indivs = np.zeros((pop_size, dim)) 

        # Loop for mutation and crossover operations
        for i in range(pop_size):

            target_v = population[i] # Target vector

            # Choice 3 random individuals from population
            indiv_mut = population[random.sample(range(pop_size), 3)]

            # Obtain F parameter value
            Fi_g = calculate_F(np.random.uniform(0.1, 1.0))

            # Determinate donor vector (mutation operation)
            donor_v = indiv_mut[0] + Fi_g * (indiv_mut[1] - indiv_mut[2])

            # Generate a random integer, which ensures that the trial vector will contain 
            # at least one individual from the donor vector
            rnd_idx = np.random.randint(1, dim)

            # Generate dim random numbers to compare with CR factor
            r_v = np.random.rand(dim)

            # Obtain CR parameter value
            CRi_g = calculate_CR(np.random.random())

            # Array to store the trial vector
            trial_v = np.zeros(dim) 
            
            # Determinate trial vector (crossover operation)
            for j in range(dim):
                if r_v[j] <= CRi_g or j == rnd_idx:
                    trial_v[j] = donor_v[j]
                elif r_v[j] > CRi_g and j != rnd_idx:
                    trial_v[j] = target_v[j]

            # If any value of the vector is out-bounds the search space, 
            # it'll be set to the corresponding limit
            if np.any(trial_v < li) or np.any(trial_v > ls):
                trial_v[trial_v < li] = li
                trial_v[trial_v > ls] = ls

            # Add the trial vector to the trial_individuals        
            trial_indivs[i] = trial_v

        # Calculate trial_individuals fitnesses
        fitness_ti = np.array([func(indiv) for indiv in trial_indivs])

        # Generate new population from fitnesess of population and trial_individuals
        population = np.array([population[i] if fitness[i] < fitness_ti[i] else trial_indivs[i] for i in range(pop_size)])

    # Obtain the best solution of the population
    optimum = population[np.argmin(np.array([func(indiv) for indiv in population]))]

    return optimum, gen


def main():
    np = 6 # Population size, must be >= 4
    dim = 2 # Dimensions
    func = sphere # Objective Function
    li, ls = -10, 10 # Limits
    gen = 30 # Generations

    optimum, gen = sade(np, dim, func, li, ls, gen)

    print(f'Optimum: {optimum} | Generations: {gen}')


if __name__ == '__main__':
    main()