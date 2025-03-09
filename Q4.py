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

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

def categorical_crossentropy(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

class FeedforwardNeuralNetwork:
    def __init__(self, input_size, output_size, momentum, beta, beta1, beta2, epsilon, config):
        self.hidden_layers = config.hidden_layers
        self.optimizer = config.optimizer
        self.epochs = config.epochs
        self.lr = config.learning_rate
        self.batch_size = config.batch_size
        self.weight_decay = config.weight_decay
        self.activation = config.activation
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        # Initialize weights & biases
        self.weights, self.biases = [], []
        self.v_w, self.v_b = [], []
        self.m_w, self.m_b = [], []

        # Input to first hidden layer
        if config.weight_initialization == 'Xavier':
            self.weights.append(np.random.randn(input_size, config.hidden_size) * np.sqrt(1 / input_size))
            self.biases.append(np.zeros((1, config.hidden_size)))
        elif config.weight_initialization == 'random':
            self.weights.append(np.random.randn(input_size, config.hidden_size) * 0.01)
            self.biases.append(np.zeros((1, config.hidden_size)))

        # Hidden layers
        for _ in range(config.hidden_layers - 1):
            if config.weight_initialization == 'Xavier':
                self.weights.append(np.random.randn(config.hidden_size, config.hidden_size) * np.sqrt(1 / config.hidden_size))
                self.biases.append(np.zeros((1, config.hidden_size)))
            elif config.weight_initialization == 'random':
                self.weights.append(np.random.randn(config.hidden_size, config.hidden_size) * 0.01)
                self.biases.append(np.zeros((1, config.hidden_size)))

        # Output layer
        if config.weight_initialization == 'Xavier':
            self.weights.append(np.random.randn(config.hidden_size, output_size) * np.sqrt(1 / config.hidden_size))
            self.biases.append(np.zeros((1, output_size)))
        elif config.weight_initialization == 'random':
            self.weights.append(np.random.randn(config.hidden_size, output_size) * 0.01)
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
        
        # For softmax + cross-entropy, the gradient at the output layer is (a_L - y)
        dz = self.a[-1] - y
        grads_w[-1] = np.dot(self.a[-2].T, dz) / m
        grads_b[-1] = np.sum(dz, axis=0, keepdims=True) / m
        
        # Backpropagate through hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            # Get the activation derivative for the current layer
            if self.activation == "sigmoid":
                da_dz = sigmoid_derivative(self.z[l])
            elif self.activation == "relu":
                da_dz = relu_derivative(self.z[l])
            elif self.activation == "tanh":
                da_dz = tanh_derivative(self.z[l])
            else:
                raise ValueError(f"Unsupported activation function: {self.activation}")
            
            # Compute gradient for current layer
            dz = np.dot(dz, self.weights[l+1].T) * da_dz
            
            # Input to the first layer is X, otherwise it's the activation from previous layer
            prev_activation = X if l == 0 else self.a[l-1]
            
            # Compute gradients for weights and biases
            grads_w[l] = np.dot(prev_activation.T, dz) / m
            grads_b[l] = np.sum(dz, axis=0, keepdims=True) / m
        
        return grads_w, grads_b


    def update_parameters(self, grads_w, grads_b):
        self.t += 1  # Update step count

        for i in range(len(self.weights)):
            if self.optimizer == "sgd":
                self.weights[i] -= self.lr * grads_w[i]
                self.biases[i] -= self.lr * grads_b[i]

            elif self.optimizer == "nag":
                # Store original parameters
                original_w = self.weights[i].copy()
                original_b = self.biases[i].copy()
                
                # Compute lookahead position
                lookahead_w = original_w + self.momentum * self.v_w[i]
                lookahead_b = original_b + self.momentum * self.v_b[i]
                
                # Temporarily set weights to lookahead for gradient computation
                self.weights[i] = lookahead_w
                self.biases[i] = lookahead_b
                
                # Compute gradients at lookahead position
                # Note: This is a simplified approach. In practice, you'd need to
                # compute gradients for all layers at the lookahead position.
                X_batch = self.a[0]  # Input data
                y_batch = self.y     # Target data (you need to store this)
                self.forward(X_batch)
                lookahead_grads_w, lookahead_grads_b = self.backward(X_batch, y_batch)
                
                # Restore original parameters
                self.weights[i] = original_w
                self.biases[i] = original_b
                
                # Update velocity using lookahead gradients
                self.v_w[i] = self.momentum * self.v_w[i] - self.lr * lookahead_grads_w[i]
                self.v_b[i] = self.momentum * self.v_b[i] - self.lr * lookahead_grads_b[i]
                
                # Apply updates
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

            elif self.optimizer == "adam":
                # Update for weights
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grads_w[i] ** 2)
                m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
                v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
                self.weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                
                # Update for biases
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads_b[i] ** 2)
                m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
                self.biases[i] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
                
            elif self.optimizer == "nadam":
                # Update for weights
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grads_w[i] ** 2)
                m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
                v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
                
                # Nadam update (includes momentum term in the update direction)
                m_w_bar = self.beta1 * m_w_hat + (1 - self.beta1) * grads_w[i] / (1 - self.beta1 ** self.t)
                self.weights[i] -= self.lr * m_w_bar / (np.sqrt(v_w_hat) + self.epsilon)
                
                # Update for biases
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads_b[i] ** 2)
                m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
                
                # Nadam update for biases
                m_b_bar = self.beta1 * m_b_hat + (1 - self.beta1) * grads_b[i] / (1 - self.beta1 ** self.t)
                self.biases[i] -= self.lr * m_b_bar / (np.sqrt(v_b_hat) + self.epsilon)

        return self.weights, self.biases


    def train(self, X_train, y_train, X_test, y_test, epochs):
        wandb.init(
            project=wandb.run.project if wandb.run else "DA6401-Assignment 1-CE21B097",
            entity=wandb.run.entity if wandb.run else "ce21b097-indian-institute-of-technology-madras",
            config={
                "optimizer": self.optimizer,
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "epochs": epochs,
                "hidden_layers": self.hidden_layers,
                "hidden_size": len(self.weights[0][0])
            },
            reinit=True  # Allow multiple runs in the same process
        )
        
        # Shuffle indices for training
        indices = np.arange(X_train.shape[0])
        
        for epoch in range(epochs):
            # Shuffle training data
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for i in range(0, X_train.shape[0], self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Store target data for NAG optimizer
                self.y = y_batch
                
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
        
        # Close the wandb run
        wandb.finish()

def train_sweep():
    # Initialize a new wandb run with config from sweep controller
    with wandb.init() as run:
        config = wandb.config
        
        # Load dataset
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train, X_test = X_train.reshape(X_train.shape[0], -1) / 255.0, X_test.reshape(X_test.shape[0], -1) / 255.0
        y_train, y_test = np.eye(10)[y_train], np.eye(10)[y_test]
        
        # Create model with sweep config parameters
        model = FeedforwardNeuralNetwork(
            input_size=X_train.shape[1],
            output_size=10,
            momentum=0.9,  # Default value, can be overridden by sweep
            beta=0.9,      # Default value, can be overridden by sweep
            beta1=0.9,     # Default value, can be overridden by sweep
            beta2=0.999,   # Default value, can be overridden by sweep
            epsilon=1e-8,  # Default value, can be overridden by sweep
            config=config
        )
        
        # Train the model
        model.train(X_train, y_train, X_test, y_test, config.epochs)

# Sweep Config
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'Validation Accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {'values': [5, 10]},
        'hidden_layers': {'values': [3, 4, 5]},
        'hidden_size': {'values': [32, 64, 128]},
        'weight_decay': {'values': [0, 0.0005, 0.5]},
        'learning_rate': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [16, 32, 64]},
        'weight_initialization': {'values': ['random', 'Xavier']},
        'activation': {'values': ['sigmoid', 'tanh', 'relu']}
    }
}

# Replace the incomplete sweep agent call with this
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
    parser.add_argument("-m", "--momentum", type=float, default=0.5)
    parser.add_argument("-beta", "--beta", type=float, default=0.5)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001)
    parser.add_argument("-run_sweep", "--run_sweep", action="store_true", help="Run hyperparameter sweep")
    args = parser.parse_args()

    if args.run_sweep:
        # Initialize the sweep
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
        # Start the sweep agent
        wandb.agent(sweep_id, function=train_sweep, count=10)
    else:
        # Load dataset
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train, X_test = X_train.reshape(X_train.shape[0], -1) / 255.0, X_test.reshape(X_test.shape[0], -1) / 255.0
        y_train, y_test = np.eye(10)[y_train], np.eye(10)[y_test]

        # Create a configuration object to match the structure expected by FeedforwardNeuralNetwork
        config = type('Config', (), {
            'hidden_layers': args.num_layers,
            'optimizer': args.optimizer,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'weight_decay': 0,  # Default value
            'activation': 'sigmoid',  # Default value
            'weight_initialization': 'Xavier',  # Default value
            'hidden_size': args.hidden_size
        })

        model = FeedforwardNeuralNetwork(
            input_size=X_train.shape[1], 
            output_size=10,
            momentum=args.momentum,
            beta=args.beta,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon,
            config=config
        )
        model.train(X_train, y_train, X_test, y_test, args.epochs)


