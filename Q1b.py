import numpy as np 

 

class Perceptron: 

    def __init__(self, num_inputs, lr=0.1): 

        self.weights = np.random.randn(num_inputs + 1) * 0.01  # Random small weights 

        self.lr = lr 

 

    def predict(self, inputs): 

        summation = np.dot(inputs, self.weights[1:]) + self.weights[0] 

        return 1 if summation > 0 else 0 

 

    def train(self, training_inputs, labels): 

        iterations = 0 

        while True: 

            error_count = 0 

            for inputs, label in zip(training_inputs, labels): 

                prediction = self.predict(inputs) 

                if prediction != label: 

                    self.weights[1:] += self.lr * (label - prediction) * inputs 

                    self.weights[0] += self.lr * (label - prediction) 

                    error_count += 1 

            iterations += 1 

            if error_count == 0: 

                break 

        return iterations 

 

# Example usage: NAND gate 

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 

labels = np.array([1, 1, 1, 0])  # NAND truth table 

 

perceptron = Perceptron(num_inputs=2) 

iterations = perceptron.train(inputs, labels) 

print("NAND converged in", iterations, "iterations") 

 

 

# XOR gate (expected to not converge) 

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 

labels = np.array([0, 1, 1, 0])  # XOR truth table 

 

perceptron = Perceptron(num_inputs=2) 

iterations = perceptron.train(inputs, labels) 

print("XOR converged in", iterations, "iterations") 

 

 

# 5-input palindrome 

def generate_palindrome_data(): 

    inputs = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], 

                       [0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 1, 0, 1, 1], [0, 1, 0, 0, 0]]) 

    labels = np.array([1, 1, 1, 1, 1, 1, 0, 0])  # Example labels for 5-input palindromes 

    return inputs, labels 

 

# 5-input majority 

def generate_majority_data(): 

    inputs = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], 

                       [0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 1, 0, 1, 1], [0, 1, 0, 0, 0]]) 

    labels = np.array([0, 0, 1, 1, 1, 0, 1, 0])  # Example labels for 5-input majority 

    return inputs, labels 

 

# 5-input parity 

def generate_parity_data(): 

    inputs = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], 

                       [0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 1, 0, 1, 1], [0, 1, 0, 0, 0]]) 

    labels = np.array([0, 0, 1, 1, 1, 1, 0, 0])  # Example labels for 5-input parity 

    return inputs, labels 

 

# Train and test on 5-input problems 

problems = { 

    'palindrome': generate_palindrome_data, 

    'majority': generate_majority_data, 

    'parity': generate_parity_data 

} 

 

for name, generate_data in problems.items(): 

    inputs, labels = generate_data() 

    perceptron = Perceptron(num_inputs=5) 

    iterations = perceptron.train(inputs, labels) 

    print(f"{name.capitalize()} converged in", iterations, "iterations") 

 segments = { 

    0: [1, 1, 1, 1, 1, 1, 0],  # 0 

} 

 

inputs = np.array([segments[0]]) 

labels = np.array([1])  # Only one digit (0), hence always 1 

 

perceptron = Perceptron(num_inputs=7) 
