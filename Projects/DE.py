import numpy as np
import random
from sklearn.datasets import load_iris

# Objective Function
def sphere(x):
    return np.sum(np.square(x))


# Function that contains differential evolution algorithm
def differential_evolution(X, y, pop_size, dim, func, li, ls, gen):
    if pop_size < 4:
        print('Population size must be >= 4')
        exit()

    # Initialize population of D-dimensions within the range
    population = np.random.uniform(li, ls, (pop_size, dim))

    # Initialize weight and crossover probability
    F = np.random.uniform(0, 2)
    CR = np.random.random()

    for _ in range(gen):

        # Evaluate individuls with the objective function and store the fitness
        fitness = np.array([func(indiv) for indiv in population], dtype=float)

        # Array to store the trial individuals after operations
        trial_indivs = np.zeros((pop_size, dim), dtype=float)
        
        # Loop for mutation and crossover operations
        for i in range(pop_size):

            target_v = population[i] # Target vector

            # Choose 2 random individuals from population
            #indiv_mut = population[random.sample(range(pop_size), 3)]

            # Determinate donor vector (mutation operation)
            #donor_v = indiv_mut[0] + F * (indiv_mut[1] - indiv_mut[2])

            random_indiv = population[random.sample(range(pop_size), 2)]
            best_indiv = population[np.argmin(fitness)]
            donor_v = F * (best_indiv - target_v) + F * (random_indiv[0] - random_indiv[1])

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
        fitness_ti = np.array([func(indiv) for indiv in trial_indivs])

        # Generate new population from fitnesess of population and trial_individuals
        population = np.array([population[i] if fitness[i] < fitness_ti[i] else trial_indivs[i] for i in range(pop_size)])

    # Obtain the best solution of the population
    optimum = population[np.argmin(np.array([func(indiv) for indiv in population]))]

    return optimum, gen


def main():
    pop_size = 10 # Population size, must be >= 4
    dim = 4 # Dimensions
    func = sphere # Objective Function
    li, ls = -10, 10 # Limits
    gen = 30 # Generations

    X, y = load_iris(return_X_y=True)

    optimum, gen = differential_evolution(X, y, pop_size, dim, func, li, ls, gen)

    print(f'Optimum: {optimum} | Generations: {gen}')

if __name__ == '__main__':
    main()