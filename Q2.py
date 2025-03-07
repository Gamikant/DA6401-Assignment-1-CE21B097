import numpy as np
import argparse
import wandb
from keras.datasets import fashion_mnist

fashion_mnist_classes = [
    "T-shirt/top",  # 0
    "Trouser",       # 1
    "Pullover",      # 2
    "Dress",         # 3
    "Coat",          # 4
    "Sandal",        # 5
    "Shirt",         # 6
    "Sneaker",       # 7
    "Bag",           # 8
    "Ankle boot"     # 9
]

# Activation functions and their derivatives
def sigmoid_neuron(x):
    return 1 / (1 + np.exp(-x))

def categorical_crossentropy(y_true, y_pred, epsilon=1e-12):

    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)  # Avoid log(0)
    loss = -np.sum(y_true * np.log(y_pred), axis=1)  # Compute loss per sample
    return np.mean(loss)  # Return mean loss over batch

# Feedforward Neural Network
class SimpleFeedforwardNeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_size, output_size):
        self.hidden_layers = hidden_layers
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_size) * 0.01)
        self.biases.append(np.random.randn(1, hidden_size))
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            self.weights.append(np.random.randn(hidden_size, hidden_size) * 0.01)
            self.biases.append(np.random.randn(1, hidden_size))
        
        # Last hidden layer to output
        self.weights.append(np.random.randn(hidden_size, output_size) * 0.01)
        self.biases.append(np.random.randn(1, output_size))

    def activate(self, x):
        return sigmoid_neuron(x)

    def forward(self, X):
        self.a = []
        self.z = []
        
        # Input layer
        self.z.append(np.dot(X, self.weights[0]) + self.biases[0])
        self.a.append(self.activate(self.z[0]))

        # Hidden layers
        for i in range(1, self.hidden_layers):
            self.z.append(np.dot(self.a[i - 1], self.weights[i]) + self.biases[i])
            self.a.append(self.activate(self.z[i]))

        # Output layer (softmax)
        self.z.append(np.dot(self.a[-1], self.weights[-1]) + self.biases[-1])
        exp_vals = np.exp(self.z[-1] - np.max(self.z[-1], axis=1, keepdims=True))
        self.a.append(exp_vals / np.sum(exp_vals, axis=1, keepdims=True))

        return self.a[-1]

    def compute_accuracy(self, X, y_true):
        y_pred = self.forward(X)
        return np.mean(np.argmax(y_pred) == np.argmax(y_true))

# Training function
def feed_forward_network(model, X, y, X_raw):
    examples = args.examples
    table = wandb.Table(columns=["Image", "Predicted Class", "Probabilities"])

    for example in range(examples):
        X_i = X[example, :].reshape(1, -1)
        y_i = y[example, :]

        # Forward pass
        prediction_probabilities = model.forward(X_i)
        predicted_class = np.argmax(prediction_probabilities)
        predicted_label = fashion_mnist_classes[predicted_class]

        # Log as a table with images
        table.add_data(
            wandb.Image(X_raw[example]),  # Log raw image
            predicted_label,  # Predicted class name
            prediction_probabilities.tolist()  # Probabilities list
        )

    # Log the full table at once
    wandb.log({"Predictions": table})

# Load Fashion-MNIST dataset
def load_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Keep a copy of raw images before normalization
    X_raw = X_train.copy()

    # Normalize and reshape
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train, X_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)

    # One-hot encode labels
    y_train_onehot = np.eye(10)[y_train]
    
    return X_train, y_train_onehot, X_raw  # Return raw images as well

# Main function
def main(args):
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config=vars(args),
    )

    # Load dataset
    X, y, X_raw= load_data()
    
    # Initialize model
    model = SimpleFeedforwardNeuralNetwork(
        input_size=X.shape[1],
        hidden_layers=args.num_layers,
        hidden_size=args.hidden_size,
        output_size=10
    )

    # Train model
    feed_forward_network(model, X, y, X_raw)

    # Finish wandb logging
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401-Assignment 1-CE21B097")
    parser.add_argument("-we", "--wandb_entity", type=str, default="ce21b097-indian-institute-of-technology-madras")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, default=4)
    parser.add_argument("-ex", "--examples", type=int, default=10)
    
    args = parser.parse_args()
    main(args)
