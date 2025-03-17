# This question is connected to Q4
from Q4 import *
import numpy as np
import argparse
import wandb
from keras.datasets import fashion_mnist, mnist
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
    parser.add_argument("-d", "--dataset", type=str, default = "fashion_mnist", choices = ["fashion_mnist", "mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-a", "--activation", type=str, default="relu", choices = ["sigmoid", "tanh", "relu"])  
    parser.add_argument("-o", "--optimizer", type=str, default="nadam", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-nhl", "--num_layers", type=int, default=3)
    parser.add_argument("-sz", "--hidden_size", type=int, default=128)
    parser.add_argument("-m", "--momentum", type=float, default=0.9)
    parser.add_argument("-beta", "--beta", type=float, default=0.9)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.99)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999)
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8)
    parser.add_argument("-wi", "--weight_initialization", type=str, default = "random", choices = ["Xavier", "random"])
    parser.add_argument("-run_sweep", "--run_sweep", action="store_true", help="Run hyperparameter sweep")
    parser.add_argument("-cnt", "--count", type=int, default = 20)
    parser.add_argument("-loss", "--loss_function", type=str, default="categorical_crossentropy", choices = ["categorical_crossentropy", "squared_error_loss"])  

    args = parser.parse_args()

    with open('best_hyperparameters.json', 'r') as f:
        best_config = json.load(f)

    if args.dataset == "mnist":
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    else:
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    val_size = 6000  # 10% of 60,000

    X_train = X_train_full[:-val_size]
    y_train = y_train_full[:-val_size]
    X_val = X_train_full[-val_size:]
    y_val = y_train_full[-val_size:]

    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_val = X_val.reshape(X_val.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    y_train = np.eye(10)[y_train]
    y_val = np.eye(10)[y_val]
    y_test = np.eye(10)[y_test]

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

    model.train(X_train, y_train, X_val, y_val, config.epochs, args.loss_function)

    y_pred = model.forward(X_test)

    test_acc = model.compute_accuracy(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    y_true_indices = np.argmax(y_test, axis=1)
    y_pred_indices = np.argmax(y_pred, axis=1)

    with wandb.init(project=args.wandb_project, entity=args.wandb_entity, name="confusion_matrix") as run:
        cm = wandb.plot.confusion_matrix(y_true=y_true_indices.tolist(), preds=y_pred_indices.tolist(), class_names=fashion_mnist_classes)
        wandb.log({"conf_mat": cm, "Test accuracy": test_acc})
        print("Heatmap confusion matrix generated and logged to wandb")

    
