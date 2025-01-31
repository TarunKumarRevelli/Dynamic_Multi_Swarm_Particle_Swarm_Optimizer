import numpy as np
import math

pi = math.pi

#Benchmark Function definations

def sphere_function(x):
    return np.sum(x**2)

def rosenbrocks_function(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackleys_function(x):
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
    term2 = -np.exp(np.mean(np.cos(2 * np.pi * x)))
    return term1 + term2 + 20 + np.e

def griewanks_function(x):
    term1 = np.sum(x**2) / 4000
    term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + term1 - term2

def rastrigins_function(x):
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

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
        
def weierstrass_function(x):
    d = 10
    s2 = d*cos_sumW2(d)
    s1 = 0
    for i in range(d):
        s1 += cos_sumW(x[i])
    return s1-s2  

#particle class with different parameters
class Particle:
    def __init__(self, dim):
        self.position = np.random.rand(dim)
        self.velocity = np.random.rand(dim)
        self.best_position = self.position.copy()
        self.fitness = float('inf')
        self.best_fitness = float('inf')

def pso_algorithm(dim, num_particles, max_iterations, c1, c2, w,func,domain):
    bounds = (np.array([domain[0]] * dim), np.array(domain[1] * dim))                #defining the position bounds (domain)
    particles = [Particle(dim) for _ in range(num_particles)]                        #initialising the particles
    global_best_position = np.zeros(dim)                                             #setting the initial global best value and position
    global_best_fitness = float('inf')

    for _ in range(max_iterations):
        for particle in particles:
            # Evaluate fitness
            particle.position = np.clip(particle.position, bounds[0], bounds[1])
            particle.fitness = func(particle.position)

            # Update personal best
            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position.copy()

            # Update global best
            if particle.best_fitness < global_best_fitness:
                global_best_fitness = particle.best_fitness
                global_best_position = particle.best_position.copy()

        for particle in particles:
            # Update velocity
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            particle.velocity = w * particle.velocity + c1 * r1 * (particle.best_position - particle.position) + c2 * r2 * (global_best_position - particle.position)

            # Update position
            particle.position = particle.position + particle.velocity

    return global_best_position, global_best_fitness

# parameters settings
dimension = 10
num_particles = 30
max_iterations = 2000  #also known as maximum genrations
c1 = 2.0  # Cognitive parameter
c2 = 2.0  # Social parameter
w = 0.5  # Inertia weight

#calling function for sphere function
best_position, best_fitness = pso_algorithm(dimension, num_particles, max_iterations, c1, c2, w,sphere_function,domain=[-5.12,5.12])

print("Sphere function:")
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)

#calling function for rosenbrocks function
best_position, best_fitness = pso_algorithm(dimension, num_particles, max_iterations, c1, c2, w,rosenbrocks_function,domain=[-2.048,2.048])

print("Rosenbrocks function:")
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)

#calling function for ackleys function
best_position, best_fitness = pso_algorithm(dimension, num_particles, max_iterations, c1, c2, w,ackleys_function,domain=[-32.768,32.768])

print("Ackley's function:")
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)

#calling function for griewanks function
best_position, best_fitness = pso_algorithm(dimension, num_particles, max_iterations, c1, c2, w,griewanks_function,domain=[-600,600])

print("Griewank's function:")
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)

#calling function for rastrigins function
best_position, best_fitness = pso_algorithm(dimension, num_particles, max_iterations, c1, c2, w,rastrigins_function,domain=[-5.12,5.12])

print("Rastrigin's function:")
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)

#calling function for weierstrass_function
best_position, best_fitness = pso_algorithm(dimension, num_particles, max_iterations, c1, c2, w,weierstrass_function,domain=[-0.5,0.5])

print("Weierstrass's function:")
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)
