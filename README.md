# COSC-5P77
Course Project

This Python code outlines the implementation and training of a simple Sum-Product Network (SPN) using PyTorch, a powerful and versatile deep learning library. The network is designed for image classification tasks, leveraging the MNIST dataset and the Fashion MNIST dataset.

_Description of Key Components:_

**1. Network Architecture (SimpleSPN class):**

Input Layer: The network begins with a linear transformation (nn.Linear) converting the flattened input images (28x28 pixels, hence 784 features) into a hidden_size dimensional space.
Sum Node: A fully connected layer (nn.Linear) that projects the input to a hidden layer of specified size (hidden_size), acting as the sum operation in SPN.
Product Node: Another fully connected layer that follows the sum node, simulating the product operations in SPNs.
Activation Function: Relu (nn.ReLU) is used after each linear transformation to introduce non-linearity.
Classifier: The final layer that maps the output of the product node to the number of classes in the dataset (10 classes).

**2. Data Preparation and Loading:**

Transforms: The input data is transformed to tensors and normalized using standard values (mean = 0.5, std = 0.5) to ensure the model receives input within a normalized range, enhancing stability during training.
Data Splitting: The datasets are randomly split into training (80%) and testing (20%) sets using SubsetRandomSampler, which helps in evaluating the model on unseen data.

**3. Training Procedure (train_spn function):**

The network is trained over multiple epochs, where in each epoch, the entire training set is passed through the network.
Loss Calculation: Cross-entropy loss (nn.CrossEntropyLoss) is used as it's suitable for classification tasks with multiple classes.
Optimization: Different optimizers like Adam and Adamax are used to update network weights based on computed gradients.
Accuracy Computation: Post each epoch, accuracy is calculated to monitor the performance of the network on the training dataset.

**4. Visualization:**

Training loss and accuracy are plotted after training with each optimizer to visually assess the training progress and compare the effectiveness of each optimizer.

**5. Execution:**

The entire training and evaluation pipeline is encapsulated in loops that handle both optimizers, demonstrating the flexibility to test multiple training strategies seamlessly.
