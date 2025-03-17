<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# 

---

# DA6401 Assignment 1: Neural Network Implementation

Repository: https://github.com/Gamikant/DA6401-Assignment-1-CE21B097
wanb report: https://wandb.ai/ce21b097-indian-institute-of-technology-madras/DA6401-Assignment%201-CE21B097/reports/CE21B097-DA6401-Assignment-1--VmlldzoxMTgxNzM0OQ?accessToken=80bfjm0sg1ranwlhzrl80liemonsjf0xrd2y3joj22hj13zbuc9sulae7x6jtsmn
This repository contains a custom implementation of a feedforward neural network for image classification on the Fashion-MNIST and MNIST datasets.

## Project Structure

- `Q1.py`: Displays sample images from each class in the Fashion-MNIST dataset
- `Q2.py`: Implements a simple feedforward neural network with random weights
- `Q3.py`: Implements backpropagation with various optimizers
- `Q4.py`: Implements hyperparameter sweeping using Weights \& Biases
- `Q7.py`: Evaluates the best model on test data and generates a confusion matrix
- `train.py`: Main script for training and evaluating the neural network
- `best_hyperparameters.json`: Stores the best hyperparameters found during sweeping


## Features

- Customizable neural network architecture (number of layers, neurons per layer)
- Multiple activation functions (sigmoid, ReLU, tanh)
- Various optimization algorithms:
    - Stochastic Gradient Descent (SGD)
    - Momentum-based gradient descent
    - Nesterov Accelerated Gradient (NAG)
    - RMSprop
    - Adam
    - Nadam
- Weight initialization methods (random, Xavier)
- Loss functions (categorical cross-entropy, squared error)
- Hyperparameter optimization using Weights \& Biases
- Experiment tracking and visualization


## Requirements

- Python 3.x
- NumPy
- Keras (for dataset loading)
- Weights \& Biases (for experiment tracking)
- Matplotlib (for visualization)


## Installation

```bash
git clone https://github.com/Gamikant/DA6401-Assignment-1-CE21B097.git
cd DA6401-Assignment-1-CE21B097
pip install numpy keras wandb matplotlib
```


## Usage

### Training with Default Parameters

```bash
python train.py
```

This will train a model on the Fashion-MNIST dataset using the default hyperparameters:

- 3 hidden layers with 128 neurons each
- ReLU activation
- Nadam optimizer
- Learning rate of 0.001
- Batch size of 32
- 20 epochs


### Customizing Training Parameters

```bash
python train.py -d mnist -e 30 -b 64 -o adam -a tanh -nhl 4 -sz 64
```

This will train a model on the MNIST dataset with:

- 4 hidden layers with 64 neurons each
- Tanh activation
- Adam optimizer
- 30 epochs
- Batch size of 64


### Running Hyperparameter Sweep

```bash
python train.py -run_sweep -d fashion_mnist -cnt 20
```

This will run a Bayesian optimization sweep with 20 trials to find the best hyperparameters for the Fashion-MNIST dataset.

### Evaluating the Best Model

```bash
python Q7.py -d mnist
```

This will evaluate the best model (loaded from `best_hyperparameters.json`) on the MNIST test set and generate a confusion matrix.

## Command-Line Arguments

The `train.py` script supports the following arguments:


| Argument | Default | Description |
| :-- | :-- | :-- |
| `-wp`, `--wandb_project` | "DA6401-Assignment 1-CE21B097" | Weights \& Biases project name |
| `-we`, `--wandb_entity` | "ce21b097-indian-institute-of-technology-madras" | Weights \& Biases entity |
| `-d`, `--dataset` | "fashion_mnist" | Dataset to use ("mnist" or "fashion_mnist") |
| `-loss`, `--loss_function` | "categorical_crossentropy" | Loss function to use |
| `-e`, `--epochs` | 20 | Number of training epochs |
| `-b`, `--batch_size` | 32 | Batch size for training |
| `-o`, `--optimizer` | "nadam" | Optimization algorithm |
| `-lr`, `--learning_rate` | 0.001 | Learning rate |
| `-m`, `--momentum` | 0.9 | Momentum for momentum-based optimizers |
| `-beta`, `--beta` | 0.9 | Beta parameter for RMSprop |
| `-beta1`, `--beta1` | 0.99 | Beta1 parameter for Adam/Nadam |
| `-beta2`, `--beta2` | 0.999 | Beta2 parameter for Adam/Nadam |
| `-eps`, `--epsilon` | 1e-8 | Epsilon for numerical stability |
| `-wi`, `--weight_initialization` | "random" | Weight initialization method |
| `-nhl`, `--num_layers` | 3 | Number of hidden layers |
| `-sz`, `--hidden_size` | 128 | Number of neurons per hidden layer |
| `-a`, `--activation` | "relu" | Activation function |
| `-run_sweep` | False | Run hyperparameter sweep |
| `-cnt`, `--count` | 20 | Number of trials for hyperparameter sweep |

## Data Handling

The code properly splits the data into training, validation, and test sets:

- Training set: 54,000 samples (first 90% of the original training data)
- Validation set: 6,000 samples (last 10% of the original training data)
- Test set: 10,000 samples (original test set)

No test data is used during training to ensure fair evaluation.

## Results

The best hyperparameters found for the Fashion-MNIST dataset are:

- 3 hidden layers with 128 neurons each
- ReLU activation
- Nadam optimizer
- Learning rate of 0.001
- Batch size of 32
- 20 epochs

These hyperparameters achieve approximately 88-89% accuracy on the Fashion-MNIST test set and 97-98% accuracy on the MNIST test set.

<div style="text-align: center">⁂</div>

[^1]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50027824/6134e699-4da2-4080-a0d4-abe52a7b5b78/CE21B097-DA6401-Assignment-1-_-DA6401-Assignment-1-CE21B097-Weights-Biases.pdf

[^2]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50027824/6ea1a0cc-7293-4158-95c4-f440ae4e5d2a/Q1.py

[^3]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50027824/fff15d59-c9c8-4802-ad0b-85cfed20120b/Q2.py

[^4]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50027824/c507d093-7c31-44a5-a3ac-24a79041eeda/Q7.py

[^5]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50027824/5fccc976-eeb0-4ec3-bb71-50bb78a3a2b9/best_hyperparameters.json

[^6]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50027824/77ce21ab-f606-4270-aa1b-17e7669f8ff3/Q3.py

[^7]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50027824/c01f3cbd-68dc-434d-8c82-6690bc874b05/Q4.py

[^8]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50027824/53908220-337e-43a7-9a6e-69391032a5fa/train.py

[^9]: https://github.com

