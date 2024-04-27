# -*- coding: utf-8 -*-
"""SPN_F-MNIST_dataset.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/185qYpKd-_MOF-BrYNwlFE_dTIsqvtU1G
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt

class SimpleSPN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleSPN, self).__init__()
        self.sum_node = nn.Linear(input_size, hidden_size)
        self.product_node = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.sum_node(x))
        x = self.relu(self.product_node(x))
        return self.classifier(x)

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).float().sum()
    return (correct / labels.size(0)) * 100

def train_spn(train_loader, model, optimizer, loss_fn, num_epochs):
    model.train()
    losses = []
    accuracies = []
    for epoch in range(num_epochs):
        total_loss = 0
        total_accuracy = 0
        for images, labels in train_loader:
            images = images.view(images.size(0), -1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels).item()

        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = total_accuracy / len(train_loader)
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}%')

    return losses, accuracies


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform) # Fashion MNIST dataset is used here

num_train = len(dataset)
indices = list(range(num_train))
split = int(np.floor(0.2 * num_train))

np.random.shuffle(indices)
train_idx, test_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=64, sampler=test_sampler)

optimizers = {
    'Adam': optim.Adam,
    'Adamax': optim.Adamax
}

for name, opt_class in optimizers.items():
    print(f"Training with {name}")
    model = SimpleSPN(28 * 28, 100, 10)
    optimizer = opt_class(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    losses, accuracies = train_spn(train_loader, model, optimizer, loss_fn, 5)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 6), losses, marker='o', label=f'{name} Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, 6), accuracies, marker='o', label=f'{name} Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')

    plt.legend()
    plt.tight_layout()
    plt.show()