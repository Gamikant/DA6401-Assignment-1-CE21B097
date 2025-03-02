import random
import argparse
import wandb

def main(args):
    print("Arguments received:")
    for arg, value in vars(args).items():
        if value is not None:  # Only print arguments that were provided
            print(f"{arg}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network with customizable hyperparameters.")

    # WandB arguments
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401-Assignment 1-CE21B097", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", type=str, default="ce21b097-indian-institute-of-technology-madras", help="WandB entity used to track experiments in Weights & Biases dashboard")

    # Dataset
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset to be used")

    # Training parameters
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs to train neural network")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size used to train neural network")

    # Loss and Optimizer
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function to be used")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd", help="Optimizer to be used")
    
    # Learning rate and optimization parameters
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1, help="Learning rate used to optimize model parameters")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum used by momentum and nag optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta used by RMSProp optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5, help="Beta1 used by Adam and Nadam optimizers")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5, help="Beta2 used by Adam and Nadam optimizers")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon used by optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay used by optimizers")

    # Network architecture
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="random", help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1, help="Number of hidden layers in the feedforward neural network")
    parser.add_argument("-sz", "--hidden_size", type=int, default=4, help="Number of hidden neurons in a feedforward layer")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid", help="Activation function for hidden layers")

    args = parser.parse_args()
    main(args)

# Initialize wandb run
run = wandb.init(
    entity="ce21b097-indian-institute-of-technology-madras",
    project="DA6401-Assignment 1-CE21B097",
    config={
        "architecture": "Feedforward Neural Network",  # Updated architecture
        "dataset": args.dataset if args.dataset else "fashion_mnist",  # Default to fashion_mnist
        "epochs": args.epochs if args.epochs else 10,  # Default to 10 epochs
        "batch_size": args.batch_size if args.batch_size else 4,  # Default to 4
        "learning_rate": args.learning_rate if args.learning_rate else 0.02,  # Default to 0.02
        "optimizer": args.optimizer if args.optimizer else "sgd",  # Default to SGD
        "loss_function": args.loss if args.loss else "cross_entropy",  # Default to cross-entropy
        "num_layers": args.num_layers if args.num_layers else 1,  # Default to 1 hidden layer
        "hidden_size": args.hidden_size if args.hidden_size else 4,  # Default to 4 neurons per layer
        "activation": args.activation if args.activation else "sigmoid",  # Default to sigmoid
    },
)

# Log all parameters to wandb
wandb.config.update(vars(args))

## Neural network training here ##

# Finish the run and upload any remaining data.
run.finish()