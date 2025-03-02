import numpy as np
import argparse
import wandb
from keras.datasets import fashion_mnist

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Loss functions
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-10  # To avoid log(0)
    return -np.sum(y_true * np.log(y_pred + epsilon)) / len(y_true)

def cross_entropy_derivative(y_true, y_pred):
    return -(y_true / (y_pred + 1e-10))

# Feedforward Neural Network
class FeedforwardNeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_size, output_size, activation):
        self.hidden_layers = hidden_layers
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_size) * 0.01)
        self.biases.append(np.zeros((1, hidden_size)))
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            self.weights.append(np.random.randn(hidden_size, hidden_size) * 0.01)
            self.biases.append(np.zeros((1, hidden_size)))
        
        # Last hidden layer to output
        self.weights.append(np.random.randn(hidden_size, output_size) * 0.01)
        self.biases.append(np.zeros((1, output_size)))

    def activate(self, x):
        if self.activation == "sigmoid":
            return sigmoid(x)
        elif self.activation == "ReLU":
            return relu(x)

    def activate_derivative(self, x):
        if self.activation == "sigmoid":
            return sigmoid_derivative(x)
        elif self.activation == "ReLU":
            return relu_derivative(x)

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

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        d_weights = [None] * len(self.weights)
        d_biases = [None] * len(self.biases)

        # Output layer gradient
        dz = cross_entropy_derivative(y, self.a[-1])
        d_weights[-1] = np.dot(self.a[-2].T, dz) / m
        d_biases[-1] = np.sum(dz, axis=0, keepdims=True) / m

        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, 0, -1):
            dz = np.dot(dz, self.weights[i + 1].T) * self.activate_derivative(self.a[i])
            d_weights[i] = np.dot(self.a[i - 1].T, dz) / m
            d_biases[i] = np.sum(dz, axis=0, keepdims=True) / m

        # First hidden layer
        dz = np.dot(dz, self.weights[1].T) * self.activate_derivative(self.a[0])
        d_weights[0] = np.dot(X.T, dz) / m
        d_biases[0] = np.sum(dz, axis=0, keepdims=True) / m

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * d_weights[i]
            self.biases[i] -= learning_rate * d_biases[i]

    def compute_accuracy(self, X, y_true):
        y_pred = self.forward(X)
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

# Training function
def train_network(model, X_train, y_train, X_val, y_val, epochs, learning_rate):
    for epoch in range(epochs):
        model.forward(X_train)
        model.backward(X_train, y_train, learning_rate)
        
        # Calculate metrics
        train_loss = cross_entropy_loss(y_train, model.forward(X_train))
        val_loss = cross_entropy_loss(y_val, model.forward(X_val))
        train_acc = model.compute_accuracy(X_train, y_train)
        val_acc = model.compute_accuracy(X_val, y_val)

        # Log metrics to wandb
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_accuracy": train_acc, "val_accuracy": val_acc})
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

# Load Fashion-MNIST dataset
def load_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Normalize and reshape
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train, X_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)

    # One-hot encode labels
    y_train_onehot = np.eye(10)[y_train]
    y_test_onehot = np.eye(10)[y_test]
    
    return X_train, y_train_onehot, X_test, y_test_onehot

# Main function
def main(args):
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config=vars(args),
    )

    # Load dataset
    X_train, y_train, X_val, y_val = load_data()
    
    # Initialize model
    model = FeedforwardNeuralNetwork(
        input_size=784,
        hidden_layers=args.num_layers,
        hidden_size=args.hidden_size,
        output_size=10,
        activation=args.activation,
    )

    # Train model
    train_network(model, X_train, y_train, X_val, y_val, args.epochs, args.learning_rate)

    # Finish wandb logging
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401-Assignment 1-CE21B097")
    parser.add_argument("-we", "--wandb_entity", type=str, default="ce21b097-indian-institute-of-technology-madras")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, default=4)
    parser.add_argument("-a", "--activation", type=str, choices=["sigmoid", "ReLU"], default="ReLU")
    
    args = parser.parse_args()
    main(args)
