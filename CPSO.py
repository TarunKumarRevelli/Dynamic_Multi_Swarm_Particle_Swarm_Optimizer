import numpy as np 

class Particle:
    def __init__(self, dim, bounds):
        # Initialize particle with a random position, velocity, and set best position and fitness to the initial position.
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = np.copy(self.position)
        self.best_fitness = float('inf')

def sphere_function(x):
    return np.sum(x**2)  # Sphere function definition.

def rosenbrock_function(x):
    # Rosenbrock function definition.
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley_function(x):
    # Ackley function definition.
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
    term2 = -np.exp(np.mean(np.cos(2 * np.pi * x)))
    return term1 + term2 + 20 + np.exp(1)

def griewank_function(x):
    # Griewank function definition.
    term1 = np.sum(x**2) / 4000
    term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + term1 - term2

def rastrigin_function(x):
    # Rastrigin function definition.
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def weierstrass_function(x, a=0.5, b=2, kmax=20):
    # Weierstrass function definition.
    term1 = np.sum([np.sum((a**k) * np.cos(2 * np.pi * b**k * (x + 0.5))) for k in range(kmax)])
    term2 = np.sum(np.fromiter(((a**k) * np.cos(2 * np.pi * b**k * 0.5) for k in range(kmax)), dtype=float))
    return abs(term1 - term2 * kmax)

def cpsp_algorithm(objective_function, dim, bounds, swarm_size=30, max_iter=2000, w=0.5, c1=1.5, c2=1.5):
    # CPSO algorithm implementation.
    particles = [Particle(dim, bounds) for _ in range(swarm_size)]  # Initialize particles with random positions and velocities.
    global_best_position = particles[0].position  # Initialize the global best position with the first particle's position.
    global_best_fitness = float('inf')  # Initialize the global best fitness value.

    for _ in range(max_iter):  # Iterate through a specified number of iterations.
        for particle in particles:
            r1, r2 = np.random.rand(dim), np.random.rand(dim) 
            # Update velocity based on the best positions and the global best position.
            particle.velocity = w * particle.velocity + c1 * r1 * (particle.best_position - particle.position) + c2 * r2 * (global_best_position - particle.position)
            # Update particle position, ensuring it stays within specified bounds.
            particle.position = np.clip(particle.position + particle.velocity, bounds[0], bounds[1])
            particle.position = np.maximum(particle.position, bounds[0])
            particle.position = np.minimum(particle.position, bounds[1])

            # Evaluate fitness of the current position.
            fitness = objective_function(particle.position)

            # Update particle's personal best if the fitness is better.
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = np.copy(particle.position)

            # Update global best if the fitness is better.
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = np.copy(particle.position)

    return global_best_position, global_best_fitness  # Return the final global best position and value.

#testing the CPSO algorithm on benchmark functions
dims = 10  #setting dimensions as per requiremnet
bounds = {                                 #setting the domains for each benchmark function
    'sphere': [-5.12, 5.12],
    'rosenbrock': [-2.048, 2.048],
    'ackley': [-32.768, 32.768],
    'griewank': [-600, 600],
    'rastrigin': [-5.12, 5.12],
    'weierstrass': [-0.5, 0.5]
}

functions = {     #creating a dictionary for the benchmark functions
    'sphere': sphere_function,
    'rosenbrock': rosenbrock_function,
    'ackley': ackley_function,
    'griewank': griewank_function,
    'rastrigin': rastrigin_function,
    'weierstrass': weierstrass_function
}

for name, bounds in bounds.items():
    #running CPSO algorithm on each benchmark function.
    result_position, result_fitness = cpsp_algorithm(functions[name], dims, bounds)
    print(f"{name.capitalize()} Function:")
    print(f"  Minimum Position: {result_position}")
    print(f"  Minimum Value: {result_fitness}\n")
