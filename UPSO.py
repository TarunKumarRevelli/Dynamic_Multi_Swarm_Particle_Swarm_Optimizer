# Unified Particle Swarm Optimization(UPSO)
import numpy as np
def sphere_function(x):
    return np.sum(x**2)

def ackley_function(x):
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.exp(1)

def rosenbrock_function(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin_function(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def griewank_function(x):
    sum1 = np.sum(x**2)
    prod1 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum1 / 4000 - prod1

def weierstrass_function(x, a=0.5, b=3, kmax=20):
    result = 0
    for k in range(kmax + 1):
        result += np.sum(a**k * np.cos(2 * np.pi * b**k * (x + 0.5)))
    return np.abs(result)

def upso_with_position(objective_function, num_particles, dimensions, bounds, max_gen, runs=20):
    global_best_values = []
    global_best_positions = []  # List to store global best positions

    for run in range(runs):
        # Initialize particles
        particles_position = np.random.uniform(bounds[0], bounds[1], size=(num_particles, dimensions))
        particles_velocity = np.zeros((num_particles, dimensions))

        # Initialize personal best positions and values
        particles_best_position = np.copy(particles_position)
        particles_best_value = np.array([objective_function(p) for p in particles_position])

        # Initialize global best position and value
        global_best_position = None
        global_best_value = np.inf

        # Initialize parameters
        w = 0.1
        Vmax = 0.1 * (bounds[1] - bounds[0])  # Reduced Vmax for stability

        for gen in range(max_gen):
            # Update global best value
            min_value = np.min(particles_best_value)
            if min_value < global_best_value:
                global_best_value = min_value
                global_best_position = particles_best_position[np.argmin(particles_best_value)]

            # Update particle velocity and position
            inertia_weight = w
            cognitive_weight = 0.9
            social_weight = 0.9

            for i in range(num_particles):
                r1 = np.random.rand(dimensions)
                r2 = np.random.rand(dimensions)

                cognitive_component = cognitive_weight * r1 * (particles_best_position[i] - particles_position[i])
                social_component = social_weight * r2 * (global_best_position - particles_position[i])

                particles_velocity[i] = inertia_weight * particles_velocity[i] + cognitive_component + social_component
                particles_velocity[i] = np.clip(particles_velocity[i], -Vmax, Vmax)
                particles_position[i] = np.clip(particles_position[i] + particles_velocity[i], bounds[0], bounds[1])

            # Update personal best values
            for i in range(num_particles):
                current_value = objective_function(particles_position[i])
                if current_value < particles_best_value[i]:
                    particles_best_value[i] = current_value
                    particles_best_position[i] = particles_position[i]

            # Update inertia weight
            w = 0.9 - (0.9 / max_gen) * gen

        # Store global best value and position
        global_best_values.append(global_best_value)
        global_best_positions.append(global_best_position)

    return global_best_values, global_best_positions

# Example usage for each function with adjusted parameters
num_particles = 30
dimensions = 10
max_gen = 2000

# Sphere function with UPSO
bounds_sphere = [-0.5, 0.5]  # Adjusted bounds for sphere function
best_values_sphere, best_positions_sphere = upso_with_position(objective_function=sphere_function, num_particles=num_particles, dimensions=dimensions, bounds=bounds_sphere, max_gen=max_gen)
print("Sphere Function - Global Best Value:", np.mean(best_values_sphere))
print("Sphere Function - Global Best Position:", np.mean(best_positions_sphere, axis=0))

# Rosenbrock's function with UPSO
bounds_rosenbrock = [-2.048, 2.048]
best_values_rosenbrock, best_positions_rosenbrock = upso_with_position(objective_function=rosenbrock_function, num_particles=num_particles, dimensions=dimensions, bounds=bounds_rosenbrock, max_gen=max_gen)
print("Rosenbrock's Function - Global Best Value:", np.mean(best_values_rosenbrock))
print("Rosenbrock's Function - Global Best Position:", np.mean(best_positions_rosenbrock, axis=0))

# Ackley's function with UPSO
bounds_ackley = [-32.678, 32.678]  # Adjusted bounds for ackley function
best_values_ackley, best_positions_ackley = upso_with_position(objective_function=ackley_function, num_particles=num_particles, dimensions=dimensions, bounds=bounds_ackley, max_gen=max_gen)
print("Ackley's Function - Global Best Value:", np.mean(best_values_ackley))
print("Ackley's Function - Global Best Position:", np.mean(best_positions_ackley, axis=0))

# Griewank's function with UPSO
bounds_griewank = [-600, 600]  # Adjusted bounds for griewank function
best_values_griewank, best_positions_griewank = upso_with_position(objective_function=griewank_function, num_particles=num_particles, dimensions=dimensions, bounds=bounds_griewank, max_gen=max_gen)
print("Griewank's Function - Global Best Value:", np.mean(best_values_griewank))
print("Griewank's Function - Global Best Position:", np.mean(best_positions_griewank, axis=0))

# Rastrigin's function with UPSO
bounds_rastrigin = [-5.12,5.12]  # Adjusted bounds for rastrigin function
best_values_rastrigin, best_positions_rastrigin = upso_with_position(objective_function=rastrigin_function, num_particles=num_particles, dimensions=dimensions, bounds=bounds_rastrigin, max_gen=max_gen)
print("Rastrigin's Function - Global Best Value:", np.mean(best_values_rastrigin))
print("Rastrigin's Function - Global Best Position:", np.mean(best_positions_rastrigin, axis=0))

# Weierstrass function with UPSO
bounds_weierstrass = [-0.5, 0.5]
best_values_weierstrass, best_positions_weierstrass = upso_with_position(objective_function=weierstrass_function, num_particles=num_particles, dimensions=dimensions, bounds=bounds_weierstrass, max_gen=max_gen)
print("Weierstrass Function - Global Best Value:", np.mean(best_values_weierstrass))
print("Weierstrass Function - Global Best Position:", np.mean(best_positions_weierstrass, axis=0))
