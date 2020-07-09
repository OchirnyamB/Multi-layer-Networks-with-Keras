# Multi-layer-Networks-with-Keras
Implementing feedforward, multi-layer networks using MNIST and CIFAR-10 with Keras library

### Requirements:
* python3
* numpy V1.19.2
* matplotlib V3.2.1
* scikit-learn V0.19.2
* Keras V2.4.3

### The goal of this repository?
> My goal is to train a neural network (using Keras) to obtain > 90% accuracy on datasets: **MNIST** and **CIFAR-10**.

### Problems to solve:
1. Classifying handwritten digits 0-9 using the MNIST Dataset
    * Fully connected 784-256-128-10 network architecture
    * Stochostic Gradient Descent Optimizer (faster convergence)
    * Softmax classifier (outputs probabilities instead of margins)
    * Mini batch size of 128 (computational speed up)
    * Epochs of 100 (number of cycle through the full training dataset)

2. Classifying an input image into 10 classes of the CIFAR-10 Dataset

### MNIST Dataset:
> The **MNIST** (“NIST” stands for National Institute of Standards and Technology while “M” stands for “modified” as the data has been preprocessed to reduce any burden on Computer Vision processing and focus solely on the task of **digit recognition**).

> Dataset consisting of 70,000 data points (7,000 examples per digit). Each data point is represented by a 784-d vector (flattened 28x28 images).

### CIFAR-10 Dataset:
> Dataset consisting of 60,000 (32x32x32 RGB images) resulting in a feature vector dimensionality of 3072. It consists of 10 classes: _airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks_.

### Evaluations of the Trained Networks:
* Handwritten Digit Recognition: [92% Accuracy on average](output/trainingEval.txt)

![kerasMNIST](/output/kerasMNIST.png)

### References:
* Deep Learning for Computer Vision with Python VOL1 by Dr.Adrian Rosebrock