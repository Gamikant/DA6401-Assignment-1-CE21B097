import numpy as np
import argparse
import wandb
from keras.datasets import fashion_mnist

fashion_mnist_classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

def categorical_crossentropy(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

class FeedforwardNeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_size, output_size, optimizer, learning_rate, batch_size, momentum, beta, beta1, beta2, epsilon):
        self.hidden_layers = hidden_layers
        self.optimizer = optimizer
        self.lr = learning_rate
        self.batch_size = batch_size
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Step counter for Adam/Nadam

        # Initialize weights & biases
        self.weights, self.biases = [], []
        self.v_w, self.v_b = [], []
        self.m_w, self.m_b = [], []

        # Input to first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_size) * 0.01)
        self.biases.append(np.zeros((1, hidden_size)))

        # Hidden layers
        for _ in range(hidden_layers - 1):
            self.weights.append(np.random.randn(hidden_size, hidden_size) * 0.01)
            self.biases.append(np.zeros((1, hidden_size)))

        # Output layer
        self.weights.append(np.random.randn(hidden_size, output_size) * 0.01)
        self.biases.append(np.zeros((1, output_size)))

        # Initialize optimizer variables
        for w, b in zip(self.weights, self.biases):
            self.v_w.append(np.zeros_like(w))
            self.v_b.append(np.zeros_like(b))
            self.m_w.append(np.zeros_like(w))
            self.m_b.append(np.zeros_like(b))

    def forward(self, X):
        self.a, self.z = [], []

        self.z.append(np.dot(X, self.weights[0]) + self.biases[0])
        self.a.append(sigmoid(self.z[0]))

        for i in range(1, self.hidden_layers):
            self.z.append(np.dot(self.a[i - 1], self.weights[i]) + self.biases[i])
            self.a.append(sigmoid(self.z[i]))

        self.z.append(np.dot(self.a[-1], self.weights[-1]) + self.biases[-1])
        self.a.append(softmax(self.z[-1]))

        return self.a[-1]

    def compute_accuracy(self, X, y):
        y_pred = self.forward(X)
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))

    def backward(self, X, y):
        m = y.shape[0]
        grads_w, grads_b = [None] * len(self.weights), [None] * len(self.biases)

        dz = self.a[-1] - y
        grads_w[-1] = np.dot(self.a[-2].T, dz) / m
        grads_b[-1] = np.sum(dz, axis=0, keepdims=True) / m

        for i in range(len(self.weights) - 2, -1, -1):
            dz = np.dot(dz, self.weights[i + 1].T) * sigmoid_derivative(self.z[i])
            grads_w[i] = np.dot((X if i == 0 else self.a[i - 1]).T, dz) / m
            grads_b[i] = np.sum(dz, axis=0, keepdims=True) / m

        return grads_w, grads_b

    def update_parameters(self, grads_w, grads_b):
        self.t += 1  # Update step count

        for i in range(len(self.weights)):
            if self.optimizer == "sgd":
                self.weights[i] -= self.lr * grads_w[i]
                self.biases[i] -= self.lr * grads_b[i]

            elif self.optimizer == "nag":  
                lookahead_w = self.weights[i] + self.momentum * self.v_w[i]
                lookahead_b = self.biases[i] + self.momentum * self.v_b[i]

                self.weights[i] = lookahead_w  # Temporarily set weights to lookahead
                self.biases[i] = lookahead_b  
                self.forward(self.a[0])  # Forward pass to get gradients
                grads_w, grads_b = self.backward(self.a[0], self.a[-1])

                self.v_w[i] = self.momentum * self.v_w[i] - self.lr * grads_w[i]
                self.v_b[i] = self.momentum * self.v_b[i] - self.lr * grads_b[i]
                self.weights[i] += self.v_w[i]
                self.biases[i] += self.v_b[i]

            elif self.optimizer == "momentum":
                self.v_w[i] = self.momentum * self.v_w[i] - self.lr * grads_w[i]
                self.v_b[i] = self.momentum * self.v_b[i] - self.lr * grads_b[i]
                self.weights[i] += self.v_w[i]
                self.biases[i] += self.v_b[i]

            elif self.optimizer == "rmsprop":
                self.v_w[i] = self.beta * self.v_w[i] + (1 - self.beta) * (grads_w[i] ** 2)
                self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * (grads_b[i] ** 2)
                self.weights[i] -= self.lr * grads_w[i] / (np.sqrt(self.v_w[i]) + self.epsilon)
                self.biases[i] -= self.lr * grads_b[i] / (np.sqrt(self.v_b[i]) + self.epsilon)

            elif self.optimizer in ["adam", "nadam"]:
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grads_w[i] ** 2)
                m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
                v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
                self.weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

    def train(self, X_train, y_train, X_test, y_test, epochs):
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "optimizer": self.optimizer,
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "epochs": epochs,
                "hidden_layers": self.hidden_layers,
                "hidden_size": len(self.weights[0][0])
            }
        )
        for epoch in range(epochs):
            for i in range(0, X_train.shape[0], self.batch_size):
                X_batch, y_batch = X_train[i:i+self.batch_size], y_train[i:i+self.batch_size]
                self.forward(X_batch)
                grads_w, grads_b = self.backward(X_batch, y_batch)
                self.update_parameters(grads_w, grads_b)

            train_loss = categorical_crossentropy(y_train, self.forward(X_train))
            val_loss = categorical_crossentropy(y_test, self.forward(X_test))
            train_acc = self.compute_accuracy(X_train, y_train)
            val_acc = self.compute_accuracy(X_test, y_test)

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

            wandb.log({
                "Epoch": epoch+1,
                "Train Loss": train_loss,
                "Validation Loss": val_loss,
                "Train Accuracy": train_acc,
                "Validation Accuracy": val_acc
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401-Assignment 1-CE21B097")
    parser.add_argument("-we", "--wandb_entity", type=str, default="ce21b097-indian-institute-of-technology-madras")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-o", "--optimizer", type=str, default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, default=4)
    parser.add_argument("-m", "--momentum", type=int, default=0.5)
    parser.add_argument("-beta", "--beta", type=int, default=0.5)
    parser.add_argument("-beta1", "--beta1", type=int, default=0.5)
    parser.add_argument("-beta2", "--beta2", type=int, default=0.5)
    parser.add_argument("-eps", "--epsilon", type=int, default=0.000001)
    args = parser.parse_args()

    # Load dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test = X_train.reshape(X_train.shape[0], -1) / 255.0, X_test.reshape(X_test.shape[0], -1) / 255.0
    y_train, y_test = np.eye(10)[y_train], np.eye(10)[y_test]

    model = FeedforwardNeuralNetwork(X_train.shape[1], args.num_layers, args.hidden_size, 10, args.optimizer, args.learning_rate, args.batch_size, args.momentum, args.beta, args.beta1, args.beta2, args.epsilon)
    model.train(X_train, y_train, X_test, y_test, args.epochs)
