import numpy as np
import math

# Constants
pi = math.pi

# Sphere function definition
def sphere_function(x):
    return np.sum(x**2)

# Rosenbrock's function definition
def rosenbrocks_function(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Ackley's function definition
def ackleys_function(x):
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
    term2 = -np.exp(np.mean(np.cos(2 * np.pi * x)))
    return term1 + term2 + 20 + np.e

# Griewank's function definition
def griewanks_function(x):
    term1 = np.sum(x**2) / 4000
    term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + term1 - term2

# Rastrigin's function definition
def rastrigins_function(x):
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

# Helper functions for Weierstrass's function
def cos_sumW(xi):
    sum = 0
    for i in range(21):
        sum += (0.5**i)*math.cos(2*pi*(3**i)*(xi+0.5))
    return sum

def cos_sumW2(d):
    sum = 0
    for i in range(21):
        sum += (0.5**i)*math.cos(2*pi*(3**i)*(0.5))
    return sum

# Weierstrass's function definition
def weierstrass_function(x):
    d = 10
    s2 = d * cos_sumW2(d)
    s1 = 0
    for i in range(d):
        s1 += cos_sumW(x[i])
    return s1 - s2

# Particle class definition
class Particle:
    def __init__(self, dim):
        self.position = np.random.rand(dim)
        self.velocity = np.random.rand(dim)
        self.best_position = self.position.copy()
        self.fitness = float('inf')
        self.best_fitness = float('inf')

# Function to regroup particles into swarms
def regroup_swarms(particles, n, m):
    # Randomly regroup particles into n swarms with m particles each
    np.random.shuffle(particles)
    return [particles[i:i+m] for i in range(0, len(particles), m)]

# Local PSO update function
def local_pso_update(particle, local_best_position, c1, c2, w):
    # Update velocity
    r1 = np.random.rand(len(particle.position))
    r2 = np.random.rand(len(particle.position))
    particle.velocity = w * particle.velocity + c1 * r1 * (particle.best_position - particle.position) + c2 * r2 * (local_best_position - particle.position)
    # Update position
    particle.position = particle.position + particle.velocity

# Global PSO update function
def global_pso_update(particle, global_best_position, c1, c2, w):
    # Update velocity
    r1 = np.random.rand(len(particle.position))
    r2 = np.random.rand(len(particle.position))
    particle.velocity = w * particle.velocity + c1 * r1 * (particle.best_position - particle.position) + c2 * r2 * (global_best_position - particle.position)
    # Update position
    particle.position = particle.position + particle.velocity

# DMS PSO algorithm
def dms_pso(dim, m, n, R, max_gen, c1, c2, w, func, domain):
    bounds = (np.array([domain[0]] * dim), np.array([domain[1]] * dim))
    particles = [Particle(dim) for _ in range(m * n)]
    swarms = regroup_swarms(particles, n, m)
    
    global_best_position = np.zeros(dim)
    global_best_fitness = float('inf')

    # Run the algorithm for the first 90% of max_gen with regrouping
    for i in range(int(0.9 * max_gen)):
        for swarm in swarms:
            local_best_position = swarm[0].best_position
            local_best_fitness = swarm[0].best_fitness

            for particle in swarm:
                # Evaluate fitness using the objective function
                particle.position = np.clip(particle.position, bounds[0], bounds[1])
                particle.fitness = func(particle.position)

                if particle.fitness < local_best_fitness:
                    local_best_fitness = particle.fitness
                    local_best_position = particle.position.copy()
                
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()

                local_pso_update(particle, local_best_position, c1, c2, w )

        if (i + 1) % R == 0:
            swarms = regroup_swarms(particles, n, m)

    # Run the algorithm for the remaining 10% of max_gen without regrouping
    for i in range(int(0.9 * max_gen), max_gen):
        for particle in particles:
            # Evaluate fitness using the objective function
            particle.position = np.clip(particle.position, bounds[0], bounds[1])
            particle.fitness = func(particle.position)

            if particle.fitness < global_best_fitness:
                global_best_fitness = particle.fitness
                global_best_position = particle.position.copy()
            
            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position.copy()

            global_pso_update(particle, global_best_position, c1, c2, w)       
                
    return global_best_position, global_best_fitness          
    
# PSO parameters
dim = 10
m = 3  # Swarm's population size
n = 10  # Number of swarms
R = 5  # Regrouping period
max_gen = 2000
c1 = 2.0  # Cognitive parameter
c2 = 2.0  # Social parameter
w = 0.5   # Inertia weight

# Run DMS PSO for various objective functions
best_position, best_fitness = dms_pso(dim, m, n, R, max_gen, c1, c2, w, sphere_function, domain=[-5.12, 5.12])
print("Sphere Function")
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)

best_position, best_fitness = dms_pso(dim, m, n, R, max_gen, c1, c2, w, rosenbrocks_function, domain=[-2.048, 2.048])
print("Rosenbrocks function:")
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)

best_position, best_fitness = dms_pso(dim, m, n, R, max_gen, c1, c2, w, ackleys_function, domain=[-32.768, 32.768])
print("Ackley's function:")
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)

best_position, best_fitness = dms_pso(dim, m, n, R, max_gen, c1, c2, w, griewanks_function, domain=[-600, 600])
print("Griewank's function:")
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)

best_position, best_fitness = dms_pso(dim, m, n, R, max_gen, c1, c2, w, rastrigins_function, domain=[-5.12, 5.12])
print("Rastrigin's function:")
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)

best_position, best_fitness = dms_pso(dim, m, n, R, max_gen, c1, c2, w, weierstrass_function, domain=[-0.5, 0.5])
print("Weierstrass's function:")
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)
