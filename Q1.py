import numpy as np
import wandb
from keras.datasets import fashion_mnist

# Initialize wandb run
wandb.init(
    entity="ce21b097-indian-institute-of-technology-madras",
    project="DA6401-Assignment 1-CE21B097",
    name = "Assignment "  # Replace with your W&B username/team
    )

# Load Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class labels in Fashion-MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Select one sample image for each class
sample_images = []
for i in range(10):  # 10 classes
    idx = np.where(train_labels == i)[0][0]  # Get first occurrence of class 'i'
    sample_images.append(train_images[idx])

# Log images to W&B
wandb_images = [wandb.Image(img, caption=class_names[i]) for i, img in enumerate(sample_images)]
wandb.log({"Fashion-MNIST Samples": wandb_images})

# Close the wandb run
wandb.finish()

