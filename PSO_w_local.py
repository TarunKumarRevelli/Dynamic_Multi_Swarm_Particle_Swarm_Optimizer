import numpy as np

def sphere_function(x):  # Definition of the sphere function.
    return np.sum(x**2)

def rosenbrocks_function(x):  # Definition of Rosenbrock's function.
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackleys_function(x):  # Definition of Ackley's function.
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
    term2 = -np.exp(np.mean(np.cos(2 * np.pi * x)))
    return term1 + term2 + 20 + np.e

def griewanks_function(x):  # Definition of Griewank's function.
    term1 = np.sum(x**2) / 4000
    term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + term1 - term2

def rastrigins_function(x):  # Definition of Rastrigin's function.
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def weierstrass_function(x, a=0.5, b=3, kmax=20):  # Definition of Weierstrass' function.
    D = len(x)
    term1 = np.sum([np.sum(np.abs(a**k * np.cos(2 * np.pi * b**k * (x + 0.5)))) for k in range(0, kmax)])
    term2 = kmax * np.sum(np.abs(a**kmax * np.cos(2 * np.pi * b**kmax * 0.5)))

    return term1 - term2

def local_pso(func, dim, swarm_size=30, max_iter=2000, w=0.5, c1=2.0, c2=2.0, domain=(-5.12, 5.12)):  # Local PSO algorithm.
    # Initialize swarm
    swarm = np.random.uniform(low=domain[0], high=domain[1], size=(swarm_size, dim))  # Randomly initialize particle positions.
    velocities = np.random.uniform(low=0, high=1, size=(swarm_size, dim))  # Randomly initialize particle velocities.
    personal_best_positions = swarm.copy()  # Set initial personal best position to current position.
    personal_best_values = np.maximum(0, np.apply_along_axis(func, 1, swarm))  # Evaluate initial fitness values.
    global_best_position = personal_best_positions[np.argmin(personal_best_values)]  # Set initial global best position.
    global_best_value = np.min(personal_best_values)  # Set initial global best value.

    for iteration in range(max_iter):  # Iterate through a specified number of iterations.
        # Update velocities and positions with inertia weight
        r1, r2 = np.random.rand(swarm_size, dim), np.random.rand(swarm_size, dim)
        velocities = w * velocities + \
                      c1 * r1 * (personal_best_positions - swarm) + \
                      c2 * r2 * (global_best_position - swarm)
        swarm = swarm + velocities  # Update particle positions.

        # Apply boundary constraints
        swarm = np.clip(swarm, domain[0], domain[1])  # Ensure particles stay within the specified domain.

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

    return global_best_position, global_best_value  # Return the final global best position and value.

#Testing the local PSO algorithm on each function
dimensions = 10  #setting dimensions as per requiremnet
print("Sphere Function:", local_pso(sphere_function, dimensions, domain=(-5.12, 5.12)))
print("Rosenbrock's Function:", local_pso(rosenbrocks_function, dimensions, domain=(-2.048, 2.048)))
print("Ackley's Function:", local_pso(ackleys_function, dimensions, domain=(-32.768, 32.768)))
print("Griewank's Function:", local_pso(griewanks_function, dimensions, domain=(-600, 600)))
print("Rastrigin's Function:", local_pso(rastrigins_function, dimensions, domain=(-5.12, 5.12)))
print("Weierstrass Function:", local_pso(weierstrass_function, dimensions, domain=(-0.5, 0.5)))
