import numpy as np

def sphere_function(x):                                                                                                #definition of sphere function
    return np.sum(x**2)

def rosenbrocks_function(x):                                                                                          #definition of rosenbrock's function
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackleys_function(x):                                                                                              #definition of ackley's function
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
    term2 = -np.exp(np.mean(np.cos(2 * np.pi * x)))
    return term1 + term2 + 20 + np.e

def griewanks_function(x):                                                                                            #definition of griewank's function
    term1 = np.sum(x**2) / 4000
    term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + term1 - term2

def rastrigins_function(x):                                                                                           #definition of rastrigin's function
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def weierstrass_function(x, a=0.5, b=3, kmax=20):                                                                     #definition of weierstrass' function
    D = len(x)
    term1 = np.sum([np.sum(np.abs(a**k * np.cos(2 * np.pi * b**k * (x + 0.5)))) for k in range(0, kmax)])
    term2 = kmax * np.sum(np.abs(a**kmax * np.cos(2 * np.pi * b**kmax * 0.5)))

    return term1 - term2

def pso(func, dim, swarm_size=30, max_iter=2000, c1=2.0, c2=2.0, constriction_factor=0.729, domain=(-5.12, 5.12)):    #main algorithm of pso
    # Initialize swarm
    swarm = np.random.uniform(low=domain[0], high=domain[1], size=(swarm_size, dim))                                  #randomly initialising positons for particles
    velocities = np.random.uniform(low=0, high=1, size=(swarm_size, dim))                                             #randomly initialising velocities for particles
    personal_best_positions = swarm.copy()                                                                            #set initial personal best position to current position
    personal_best_values = np.maximum(0, np.apply_along_axis(func, 1, swarm))                                         #evaluate initial fitness values
    global_best_position = personal_best_positions[np.argmin(personal_best_values)]                                   #set initial global best position
    global_best_value = np.min(personal_best_values)                                                                  #set initial global best value

    for iteration in range(max_iter): #running the algo until max no. of iterations is reached
        # Update velocities and positions
        r1, r2 = np.random.rand(swarm_size, dim), np.random.rand(swarm_size, dim)
        velocities = constriction_factor * (velocities +
                                           c1 * r1 * (personal_best_positions - swarm) +
                                           c2 * r2 * (global_best_position - swarm))
        swarm = swarm + velocities

        # Apply boundary constraints
        swarm = np.clip(swarm, domain[0], domain[1])

        # Update personal best
        current_values = np.maximum(0, np.apply_along_axis(func, 1, swarm))
        update_mask = current_values < personal_best_values
        personal_best_positions[update_mask] = swarm[update_mask]
        personal_best_values[update_mask] = current_values[update_mask]

        # Update global best
        min_index = np.argmin(personal_best_values)
        if personal_best_values[min_index] < global_best_value:
            global_best_position = personal_best_positions[min_index]
            global_best_value = personal_best_values[min_index]

    return global_best_position, global_best_value

#testing the PSO algorithm on each function in their respective domains
dimensions = 10  #setting the dimensions as per the problem's requirement
print("Sphere Function:", pso(sphere_function, dimensions, domain=(-5.12, 5.12)))
print("Rosenbrock's Function:", pso(rosenbrocks_function, dimensions, domain=(-2.048, 2.048)))
print("Ackley's Function:", pso(ackleys_function, dimensions, domain=(-32.768, 32.768)))
print("Griewank's Function:", pso(griewanks_function, dimensions, domain=(-600, 600)))
print("Rastrigin's Function:", pso(rastrigins_function, dimensions, domain=(-5.12, 5.12)))
print("Weierstrass Function:", pso(weierstrass_function, dimensions, domain=(-0.5, 0.5)))
