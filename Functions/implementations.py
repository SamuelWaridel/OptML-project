# .py file for different functions used in then notebook

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, recall_score, f1_score
from IPython.display import clear_output
import torchvision.models as models
import random

class SimpleCNN(nn.Module):
    '''A simple CNN model for CIFAR-10 classification.'''
    def __init__(self):
        '''Initialize the model.
        The model consists of two convolutional layers followed by two fully connected layers.
        The convolutional layers use ReLU activation and max pooling.
        The fully connected layers use ReLU activation.
        '''
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        '''Forward pass through the model.
        '''
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x
    
class VGGLike(nn.Module):
    '''A VGG-like model for CIFAR-10 classification.'''
    def __init__(self):
        '''Initialize the model.
        The model consists of three convolutional layers followed by two fully connected layers.
        The convolutional layers use ReLU activation and max pooling.
        The fully connected layers use ReLU activation.
        '''
        super(VGGLike, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 32 → 16

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16 → 8

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 8 → 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        '''Forward pass through the model.
        '''
        x = self.features(x)
        x = self.classifier(x)
        return x
    
def get_resnet18_cifar():
    '''Get a modified ResNet-18 model for CIFAR-10 classification.
    The model is modified to accept 32x32 images and has 10 output classes.
    The first convolutional layer is modified to accept 3 input channels.
    The max pooling layer is removed to preserve spatial resolution.
    '''
    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove the first maxpool to preserve spatial resolution
    return model
    
def get_densenet121():
    model = models.densenet121(num_classes=10)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.features.pool0 = nn.Identity()
    return model

def get_data_loaders(batch_size=64, valid_size = 0.1, random_seed = 42):
    '''Get CIFAR-10 data loaders for training and testing.
    The data is normalized using the CIFAR-10 mean and std.
    The training data is split into training and validation sets.
    
    Parameters:
        batch_size (int): Batch size for data loaders.
        valid_size (float): Proportion of training data to use for validation.
        random_seed (int): Seed for random number generation.
    
    Returns:
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for testing data.
    '''
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    split = int(np.floor(valid_size * len(train_dataset)))
    indices = list(range(len(train_dataset)))
    train_idx, valid_idx = indices[split:], indices[:split] 

    generator = torch.Generator() # Create a generator for reproducibility
    generator.manual_seed(random_seed) # Set the seed for the generator

    train_sampler = SubsetRandomSampler(train_idx, generator)
    valid_sampler = SubsetRandomSampler(valid_idx, generator)

    train_loader = DataLoader(train_dataset, sampler = train_sampler, batch_size=batch_size)
    valid_loader = DataLoader(train_dataset, sampler = valid_sampler, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def evaluate_model(model, dataloader, device):
    '''Evaluate the model on the test data.
    The model is set to evaluation mode.
    The predictions are made on the test data.
    The accuracy, recall, and F1 score are calculated.
    '''
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, recall, f1

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    '''Train the model for one epoch.
    The model is set to training mode.
    The optimizer and loss function are used to update the model weights.
    The loss is calculated and backpropagated.
    '''
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def train(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=5):
    '''Train the model for a specified number of epochs.
    The model is trained on the training data and evaluated on the test data.
    The accuracy, recall, and F1 score are printed after each epoch.
    '''
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc, rec, f1 = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Accuracy: {acc:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
def train_and_return_evaluation_SGD(model_fn, lr, momentum, train_loader, valid_loader, test_loader, device, epochs=100, eval_interval=10):
    '''Train the model using SGD optimizer and return evaluation metrics on the validation set.
    The model is trained for a specified number of epochs.
    The accuracy, recall, and F1 score are printed after each evaluation interval.
    
    Args:
        model_fn: Function to create the model.
        lr: Learning rate for the optimizer.
        momentum: Momentum for the optimizer.
        train_loader: DataLoader for training data.
        valid_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
        device: Device to run the model on (CPU or GPU).
        epochs: Number of epochs to train the model.
        eval_interval: Interval for evaluating the model.
    Returns:
        scores: List of tuples containing epoch, accuracy, recall, and F1 score.
        model: Trained model.
    '''
    clear_output(wait=True)

    model = model_fn().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    scores = []

    print(f"\n🔧 Training with SGD: lr={lr}, momentum={momentum}")
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        if epoch % eval_interval == 0:
            acc, rec, f1 = evaluate_model(model, valid_loader, device)
            print(f"Epoch {epoch} | Acc={acc:.4f} | Recall={rec:.4f} | F1={f1:.4f}")
            scores.append((epoch, acc, rec, f1))
            
    # Evaluate on the test set after training
    acc, rec, f1 = evaluate_model(model, test_loader, device)
    scores.append(("Test", acc, rec, f1))
    print(f"\nTest Set Evaluation: Acc={acc:.4f} | Recall={rec:.4f} | F1={f1:.4f}")

    return scores, model

def set_seed(seed=42):
    '''Set the seed for random number generation.
    This ensures that the results are reproducible.
    
    Args:
        seed: Seed value for random number generation.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # Makes results reproducible
    torch.backends.cudnn.benchmark = False     # Slower, but deterministic
    
def train_and_return_evaluation_Adam(model_fn, lr, betas, train_loader, valid_loader, test_loader, device, epochs=100, eval_interval=10):
    '''Train the model using SGD optimizer and return evaluation metrics.
    The model is trained for a specified number of epochs.
    The accuracy, recall, and F1 score are printed after each evaluation interval.
    
    Args:
        model_fn: Function to create the model.
        lr: Learning rate for the optimizer.
        betas: Betas for the optimizer.
        train_loader: DataLoader for training data.
        valid_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
        device: Device to run the model on (CPU or GPU).
        epochs: Number of epochs to train the model.
        eval_interval: Interval for evaluating the model.
    Returns:
        scores: List of tuples containing epoch, accuracy, recall, and F1 score.
        model: Trained model.
    '''
    clear_output(wait=True)

    model = model_fn().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas = betas)
    criterion = nn.CrossEntropyLoss()

    scores = []

    print(f"\n🔧 Training with Adam: lr={lr}, betas={betas}")
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        if epoch % eval_interval == 0:
            acc, rec, f1 = evaluate_model(model, valid_loader, device)
            print(f"Epoch {epoch} | Acc={acc:.4f} | Recall={rec:.4f} | F1={f1:.4f}")
            scores.append((epoch, acc, rec, f1))

    # Evaluate on the test set after training
    acc, rec, f1 = evaluate_model(model, test_loader, device)
    scores.append(("Test", acc, rec, f1))
    print(f"\nTest Set Evaluation: Acc={acc:.4f} | Recall={rec:.4f} | F1={f1:.4f}")

    return scores, model

def train_and_return_evaluation_Adagrad(model_fn, lr, lr_decay, weight_decay, train_loader, valid_loader, test_loader, device, epochs=100, eval_interval=10):
    '''
    Train the model using Adagrad optimizer and return evaluation metrics.
    
    Args:
        model_fn: Function to create the model.
        lr: Learning rate for Adagrad.
        train_loader: Training data loader.
        valid_loader: Validation data loader.
        test_loader: Test data loader.
        device: "cuda" or "cpu".
        epochs: Number of training epochs.
        eval_interval: Evaluate every `eval_interval` epochs.
        weight_decay: L2 regularization parameter.
        scheduler_type: Optional scheduler ("StepLR", "ExponentialLR", etc.).
        scheduler_kwargs: Dict of kwargs to initialize the scheduler.
    
    Returns:
        scores: List of tuples (epoch, acc, rec, f1).
        model: Trained model.
    '''

    #clear_output(wait=True)

    model = model_fn().to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    scores = []

    print(f"\n🔧 Training with Adagrad: lr={lr}, lr_decay={lr_decay}, weight_decay={weight_decay}")
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        if epoch % eval_interval == 0:
            acc, rec, f1 = evaluate_model(model, valid_loader, device)
            print(f"Epoch {epoch} | Acc={acc:.4f} | Recall={rec:.4f} | F1={f1:.4f}")
            scores.append((epoch, acc, rec, f1))
    
    # Test set evaluation
    acc, rec, f1 = evaluate_model(model, test_loader, device)
    scores.append(("Test", acc, rec, f1))
    print(f"\nTest Set Evaluation: Acc={acc:.4f} | Recall={rec:.4f} | F1={f1:.4f}")

    return scores, model
