import numpy as np

# Sphere function
def sphere(individual):
    return np.sum(np.square(individual))

# Fitness function
def fitness_func(i_fitness, w_fitness):
    return i_fitness - w_fitness + 1

# Function containing the bumda algorithm
def bumda(pop_size, dim, func, t_value, li, ls, gen=0, min_stdev=0):

    # Initialize population of D-dimensions within the range
    population = np.random.uniform(li, ls, (pop_size, dim))

    # Evaluate the population up to the "minstd" value or the number of "gen" reached.
    #for _ in range(gen):
    while True:
        gen += 1

        # Evaluate individuals with the objective function and store the fitness
        fitness = np.array([func(indiv) for indiv in population])

        # Obtain the t-first indexes with the best fitness
        t_idx = fitness.argsort()[:t_value]
        
        # Truncate the t-first individuals with the best fitness
        t_population = population[t_idx]

        # Store the t-first fitness values
        t_fitness = fitness[t_idx]
        
        # Store the worst fitness value of the truncated population
        w_fitness = fitness[t_idx[-1]]

        # Compute mean of the truncated population
        num_sum, den_sum = 0, 0
        for i, indiv in enumerate(t_population):
            num_sum += indiv * fitness_func(t_fitness[i], w_fitness)
            den_sum += fitness_func(t_fitness[i], w_fitness)

        mean = num_sum / den_sum # Mean value

        # Compute variance of the truncated population
        num_sum, den_sum = 0, 0
        for i, indiv in enumerate(t_population):
            num_sum += fitness_func(t_fitness[i], w_fitness) * np.square((indiv - mean))
            den_sum += fitness_func(t_fitness[i], w_fitness)

        variance = num_sum / (1 + den_sum) # Variance value

        # If any value of the variance is negative, it is multiplied by -1
        if np.any(variance < 0): 
            variance[variance < 0] *= -1

        std = np.sqrt(variance) # Standard deviation
        
        # Stop criteria: the standard deviation converges to the minimum accepted value
        if std.all() < min_stdev:
            print(f'Mininum standard deviation value reached: {min_stdev}')
            break
        
        # Generate new population based on mean and standard deviation
        population = np.random.normal(mean, std, ((pop_size-1), dim))

        # Add the elite individual (the best evaluated) to the new population
        population = np.reshape(np.append(population, t_population[0]), (pop_size, dim))

    return t_population[0], gen

def main():
    nsample = 10 # Population size
    dim = 2 # Dimensions
    func = sphere # Objective function
    li, ls = -10, 5 # Limits
    t_value = 3 # Truncation value

    ''' Comentar la línea 18, y descomentar la 19 y 20 antes de ejecutar las siguientes 2 líneas'''
    min_stdev = 0.0000007 # Minimun standard deviation value
    optimum, gen = bumda(nsample, dim, func, t_value, li, ls, min_stdev=min_stdev)
    
    ''' Comentar la línea 19 y 20, y descomentar la 18 antes de ejecutar las siguientes 2 líneas'''
    #gen = 100 # Number of generations
    #optimum, gen = bumda(nsample, dim, func, t_value, li, ls, gen=gen)

    print(f'Optimum: {optimum} | Generations: {gen}')


if __name__ == "__main__":
    main()