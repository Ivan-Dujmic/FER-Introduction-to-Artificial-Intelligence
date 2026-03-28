import csv
import sys
import numpy as np
import random

def load_data(path):
    with open(path, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)    # We don't need the header
        data = [list(map(float, row)) for row in reader]
    return np.array(data)

class NeuralNetwork:
    ACTIVATION_FUNCTIONS = {
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
    }

    def __init__(self, architecture, input_size, output_size, activation_function='sigmoid'):
        self.input_size = input_size
        self.output_size = output_size
        # Architecture example "3s5s" meaning 2 hidden layers with 3 and 5 neurons
        self.layers = [int(x) for x in architecture.split('s') if x]
        self.layers.append(output_size)  # Add output layer size
        self.activation = self.ACTIVATION_FUNCTIONS[activation_function]

        self.weights = []
        self.biases = []

        # Weight and bias initialization parameters
        self.init_mean = 0.0
        self.init_std_dev = 0.01

    def _initialize_population(self, pop_size):
        population = []
        for _ in range(pop_size):
            weights = []
            biases = []
            layer_input_size = self.input_size
            for layer_size in self.layers:
                weights.append(np.random.randn(layer_input_size, layer_size) * self.init_std_dev + self.init_mean)
                biases.append(np.random.randn(layer_size) * self.init_std_dev + self.init_mean)
                layer_input_size = layer_size
            population.append((weights, biases))
        return population

    def evaluate(self, X_init, Y, populations=None):
        if populations is None:
            populations = [(self.weights, self.biases)]

        pop_size = len(populations)
        X = np.expand_dims(X_init, 0)  # Shape: (1, N, D)
        X = np.repeat(X, pop_size, axis=0)  # Shape: (P, N, D)

        out = X
        for layer_idx in range(len(self.layers)):
            weights = np.stack([p[0][layer_idx] for p in populations])  # (P, D_in, D_out)
            biases = np.stack([p[1][layer_idx] for p in populations])   # (P, D_out)

            out = np.matmul(out, weights) + biases[:, np.newaxis, :]  # (P, N, D_out)

            if layer_idx < len(self.layers) - 1:
                out = self.activation(out)

        errors = np.mean(np.sum((Y[np.newaxis, :, :] - out) ** 2, axis=2), axis=1)
        return errors
    
    def _crossover_and_mutate(self, parent1, parent2, mutation_prob, mutation_std_dev):
        child_weights = []
        child_biases = []

        for w1, w2 in zip(parent1[0], parent2[0]):
            child = (w1 + w2) / 2
            if random.random() < mutation_prob:
                child += np.random.randn(*child.shape) * mutation_std_dev
            child_weights.append(child)

        for b1, b2 in zip(parent1[1], parent2[1]):
            child = (b1 + b2) / 2
            if random.random() < mutation_prob:
                child += np.random.randn(*child.shape) * mutation_std_dev
            child_biases.append(child)

        return (child_weights, child_biases)

    def train(self, data, iterations, pop_size, elitism, mutation_prob, mutation_std_dev):
        population = self._initialize_population(pop_size)

        best_err = float('inf')

        X = data[:, :-self.output_size]  # All rows, all columns except last output columns
        Y = data[:, -self.output_size:]  # Last output_size columns

        errors = self.evaluate(X, Y, population)
        sorted_indices = np.argsort(errors)

        for iteration in range(iterations):
            new_population = [population[i] for i in sorted_indices[:elitism]]  # Select the best individuals
            fitness = np.array(1 / (np.array(errors) + 1e-4))   # Avoid division by zero
            fitness_sum = fitness.sum()
            while len(new_population) < pop_size:
                # Select two parents randomly proportional to their fitness
                random_indices = np.random.choice(pop_size, size=2, replace=False, p=fitness/fitness_sum)
                parent1 = population[random_indices[0]]
                parent2 = population[random_indices[1]]
                child_weights, child_biases = self._crossover_and_mutate(parent1, parent2, mutation_prob, mutation_std_dev)
                new_population.append((child_weights, child_biases))
                
            population = new_population

            errors = self.evaluate(X, Y, population)

            sorted_indices = np.argsort(errors)
            min_err = errors[sorted_indices[0]]
            if min_err < best_err:
                best_err = min_err
                best_unit = population[sorted_indices[0]]
                self.weights = [w.copy() for w in best_unit[0]]
                self.biases = [b.copy() for b in best_unit[1]]

            if (iteration + 1) % 2000 == 0:
                print(f"[Train error @{iteration + 1}]: {best_err:.6f}")

def main():
    if "--train" in sys.argv:
        train_file = sys.argv[sys.argv.index("--train") + 1]
    if "--test" in sys.argv:
        test_file = sys.argv[sys.argv.index("--test") + 1]
    if "--nn" in sys.argv:
        architecture = sys.argv[sys.argv.index("--nn") + 1]
    if "--popsize" in sys.argv:
        pop_size = int(sys.argv[sys.argv.index("--popsize") + 1])
    if "--elitism" in sys.argv:
        elitism = int(sys.argv[sys.argv.index("--elitism") + 1])
    if "--p" in sys.argv:
        mutation_prob = float(sys.argv[sys.argv.index("--p") + 1])
    if "--K" in sys.argv:
        mutation_std_dev = float(sys.argv[sys.argv.index("--K") + 1])
    if "--iter" in sys.argv:
        iterations = int(sys.argv[sys.argv.index("--iter") + 1])

    train_data = load_data(train_file)
    test_data = load_data(test_file)

    nn = NeuralNetwork(architecture, train_data.shape[1] - 1, 1)
    nn.train(train_data, iterations, pop_size, elitism, mutation_prob, mutation_std_dev)
    test_error = nn.evaluate(test_data[:, :-1], test_data[:, -1:])[0]
    print(f"[Test error]: {test_error:.6f}")

if __name__ == "__main__":
    main()