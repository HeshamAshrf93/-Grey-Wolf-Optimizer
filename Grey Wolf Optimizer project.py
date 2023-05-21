import numpy as np

class GreyWolfOptimizer:
    def __init__(self, objective_function, num_dimensions, search_space, max_iterations, population_size):
        self.objective_function = objective_function
        self.num_dimensions = num_dimensions
        self.search_space = search_space
        self.max_iterations = max_iterations
        self.population_size = population_size

    def optimize(self):
        alpha_pos = np.zeros(self.num_dimensions)
        beta_pos = np.zeros(self.num_dimensions)
        delta_pos = np.zeros(self.num_dimensions)

        alpha_score = float('inf')
        beta_score = float('inf')
        delta_score = float('inf')

        # Initialize the population
        population = np.zeros((self.population_size, self.num_dimensions))
        for i in range(self.population_size):
            population[i, :] = self.search_space[0] + (self.search_space[1] - self.search_space[0]) * np.random.rand(self.num_dimensions)

        iteration = 0
        while iteration < self.max_iterations:
            for i in range(self.population_size):
                # Calculate fitness for each individual
                fitness = self.objective_function(population[i, :])

                # Update alpha, beta, and delta positions and scores
                if fitness < alpha_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = alpha_score
                    beta_pos = alpha_pos.copy()
                    alpha_score = fitness
                    alpha_pos = population[i, :].copy()
                elif fitness < beta_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = fitness
                    beta_pos = population[i, :].copy()
                elif fitness < delta_score:
                    delta_score = fitness
                    delta_pos = population[i, :].copy()

            a = 2 - iteration * ((2) / self.max_iterations)  # Parameter a decreases linearly from 2 to 0

            # Update the positions of grey wolves
            for i in range(self.population_size):
                for j in range(self.num_dimensions):
                    r1 = np.random.rand()  # Randomization parameter
                    r2 = np.random.rand()  # Randomization parameter

                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2

                    D_alpha = abs(C1 * alpha_pos[j] - population[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha

                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2

                    D_beta = abs(C2 * beta_pos[j] - population[i, j])
                    X2 = beta_pos[j] - A2 * D_beta

                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2

                    D_delta = abs(C3 * delta_pos[j] - population[i, j])
                    X3 = delta_pos[j] - A3 * D_delta

                    population[i, j] = (X1 + X2 + X3) / 3  # Update the position of the current wolf

            iteration += 1

        return alpha_pos, alpha_score

# Example usage of the Grey Wolf Optimizer for a different optimization problem

def custom_objective(x):
    # Define your custom objective function
    return np.sum(np.square(x))  # Example: minimize the sum of squared elements

def main():
    # Define the problem
    num_dimensions = 10
    search_space = (-5, 5)  # Search space for the problem
    max_iterations = 100
    population_size = 50

    # Initialize the Grey Wolf Optimizer
    gwo = GreyWolfOptimizer(custom_objective, num_dimensions, search_space, max_iterations, population_size)

    # Optimize the problem using GWO
    best_solution, best_score = gwo.optimize()

    print("Best solution:", best_solution)
    print("Best score:", best_score)

if __name__ == '__main__':
    main()
