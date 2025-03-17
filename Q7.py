# This question is connected to Q4
from Q4 import *
import numpy as np
import argparse
import wandb
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import os
import json

fashion_mnist_classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401-Assignment 1-CE21B097")
    parser.add_argument("-we", "--wandb_entity", type=str, default="ce21b097-indian-institute-of-technology-madras")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-a", "--activation", type=str, default="sigmoid", choices = ["sigmoid", "tanh", "relu"])  
    parser.add_argument("-o", "--optimizer", type=str, default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, default=4)
    parser.add_argument("-m", "--momentum", type=float, default=0.5)
    parser.add_argument("-beta", "--beta", type=float, default=0.5)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001)

    args = parser.parse_args()

    with open('best_hyperparameters.json', 'r') as f:
        best_config = json.load(f)

    # Load the original train/test split
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    # Define the validation set size (e.g., 10% of the training data)
    val_size = 6000  # 10% of 60,000

    # Split the training data into training and validation sets
    X_train = X_train_full[:-val_size]
    y_train = y_train_full[:-val_size]
    X_val = X_train_full[-val_size:]
    y_val = y_train_full[-val_size:]

    # Reshape and normalize
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_val = X_val.reshape(X_val.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    # One-hot encode the labels
    y_train = np.eye(10)[y_train]
    y_val = np.eye(10)[y_val]
    y_test = np.eye(10)[y_test]

    # Create a configuration object to match the structure expected by FeedforwardNeuralNetwork
    config = type('Config', (), best_config)

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

    # Train the model
    model.train(X_train, y_train, X_val, y_val, args.epochs)

    # Get predictions for test set
    y_pred = model.forward(X_test)

    test_acc = model.compute_accuracy(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # After getting predictions and converting to indices
    y_true_indices = np.argmax(y_test, axis=1)
    y_pred_indices = np.argmax(y_pred, axis=1)

    # # Create confusion matrix manually
    # cm = np.zeros((10, 10), dtype=int)
    # for i, j in zip(y_true_indices, y_pred_indices):
    #     cm[i, j] += 1

    # Start a new wandb run for the confusion matrix
    with wandb.init(project=args.wandb_project, entity=args.wandb_entity, name="confusion_matrix") as run:
        # Use wandb.plots.HeatMap instead of confusion_matrix
        cm = wandb.plot.confusion_matrix(y_true=y_true_indices.tolist(), preds=y_pred_indices.tolist(), class_names=fashion_mnist_classes)
        wandb.log({"conf_mat": cm})
        print("Heatmap confusion matrix generated and logged to wandb")

    
