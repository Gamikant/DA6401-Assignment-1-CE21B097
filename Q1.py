import numpy as np
import wandb
from keras.datasets import fashion_mnist

# Initializing wandb project where the experiment tracking will be done
wandb.init(
    entity="ce21b097-indian-institute-of-technology-madras",
    project="DA6401-Assignment 1-CE21B097",
    name = "Question 1 " 
    )

# Loading fashion_mnist data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Displaying 1 image from each class
sample_images = []
for i in range(10): 
    idx = np.where(train_labels == i)[0][0]
    sample_images.append(train_images[idx])

wandb_images = [wandb.Image(img, caption=class_names[i]) for i, img in enumerate(sample_images)]
wandb.log({"Fashion-MNIST Samples": wandb_images})

wandb.finish()

