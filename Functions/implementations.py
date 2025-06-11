"""This module contains various functions for training, evaluating, and attacking models on the CIFAR-10 dataset.
It includes functions for model definition, data loading, training, evaluation, and black-box attacks using Foolbox."""

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
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, f1_score
from IPython.display import clear_output
import torchvision.models as models
import random
import foolbox as fb
from foolbox.attacks import BoundaryAttack
from tqdm import tqdm
import os
import requests
import tarfile

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
    
def train_and_return_evaluation_Adam(model_fn, lr, beta_1, beta_2, train_loader, valid_loader, test_loader, device, epochs=100, eval_interval=10):
    '''Train the model using SGD optimizer and return evaluation metrics.
    The model is trained for a specified number of epochs.
    The accuracy, recall, and F1 score are printed after each evaluation interval.
    
    Args:
        model_fn: Function to create the model. Options include SimpleCNN, VGGLike, get_resnet18_cifar, get_densenet121.
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
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta_1, beta_2))
    criterion = nn.CrossEntropyLoss()

    scores = []

    print(f"\n🔧 Training with Adam: lr={lr}, beta_1={beta_1}, beta_2={beta_2}")
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
        model_fn: Function to create the model. Options include SimpleCNN, VGGLike, get_resnet18_cifar, get_densenet121.
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
            
    # Evaluate on the test set after training
    acc, rec, f1 = evaluate_model(model, test_loader, device)
    scores.append(("Test", acc, rec, f1))
    print(f"\nTest Set Evaluation: Acc={acc:.4f} | Recall={rec:.4f} | F1={f1:.4f}")

    return scores, model

def extract_params_Adam(model_name):
    """General quality of life function to extract the parameters from the model name.
    This function assumes the model name is formatted as 'ModelType_lr_LearningRate_beta1_Beta1Value_beta2_Beta2Value.pth'.

    Args:
        model_name (str): The name of the model file.

    Returns:
        learning_rate (float): The learning rate extracted from the model name.
        beta_1 (float): The beta_1 value extracted from the model name.
        beta_2 (float): The beta_2 value extracted from the model name.
    """
    parts = model_name.split('_') # Split the model name by underscores
    learning_rate = float(parts[2])
    beta_1 = float(parts[4]) 
    beta_2 = float(parts[6][:-4])
    return learning_rate, beta_1, beta_2

def get_best_models(list_best_models, best_models_dir, device):
    """
    Function to retrieve the best models from a given list.
    
    Args:
        list_best_models (list): List of model names to retrieve.
    
    Returns:
        dict: Dictionary containing the loaded models.
    """
    best_models = {}
    for model_name in list_best_models:
        model_path = os.path.join(best_models_dir, model_name) # Construct the full path to the model file
        
        # Check if the model file exists
        if os.path.exists(model_path):
            parts = model_name.split('_') # Split the model name by underscores, to extract the model type
            
            # Initialize the model based on its type
            if 'VGG' in parts[1]:
                model = VGGLike().to(device)  # for vgg models
            elif 'ResNet' in parts[1]:
                model = get_resnet18_cifar().to(device)
            elif 'DenseNet' in parts[1]:
                model = get_densenet121().to(device)  # for densenet models
            else:
                print(f"Unknown model type in {model_name}. Skipping.")
                continue
            
            # Load the model state dictionary
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()  # Set the model to evaluation mode
            
            best_models[parts[1]] = model # Store the model in the dictionary with its type as the key
        else:
            print(f"Model {model_name} not found in {best_models_dir}")
    return best_models

def download_cifar10c_folder(download_dir='data'):
    """
    Downloads and extracts the entire CIFAR-10-C dataset from Zenodo if not already present.
    
    Args:
        download_dir (str, optional): The directory where the CIFAR-10-C dataset will be downloaded. Defaults to 'data'.
    
    Returns:
        str: The path to the downloaded tar file.
    """
    
    os.makedirs(download_dir, exist_ok=True)
    tar_path = os.path.join(download_dir, "CIFAR-10-C.tar")

    # Download the .tar file from the CIFAR-10-C github repository
    if not os.path.exists(tar_path):
        url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
        print(f"Downloading CIFAR-10-C dataset from {url} ...")
        response = requests.get(url, stream=True) # download the file
        if response.status_code == 200: # check if the request was successful
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): # write the file in chunks
                    f.write(chunk)
            print(f"Download complete: {tar_path}")
        else:
            raise RuntimeError(f"Failed to download dataset (status code: {response.status_code})")
    return tar_path

def extract_cifar10c_corruption(tar_path = 'data/CIFAR-10-C.tar'):
    """ Extracts the CIFAR-10-C dataset from the downloaded tar file.

    Args:
        tar_path (str, optional): The file path to the downloaded tar file. Defaults to 'data/CIFAR-10-C.tar'.

    Returns:
        extract_folder: The path to the extracted CIFAR-10-C dataset folder.
    """
    
    download_dir = os.path.dirname(tar_path)  # Get the directory where the tar file is located
    extract_folder = os.path.join(download_dir, "CIFAR-10-C")
    
    # Extract the .tar file
    if not os.path.exists(extract_folder):
        print(f"Extracting {tar_path} ...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=download_dir, filter="data")
        print(f"Extraction complete.")

    return extract_folder

def evaluate_model_on_corruption(corruption_type, severity, model):
    """
    Evaluate the model on a specific corruption type and severity level from CIFAR-10-C dataset.
    Args:
        corruption_type (str): The type of corruption to evaluate (e.g., 'gaussian_noise').
        severity (int): The severity level of the corruption (1 to 5).
        model: The model to evaluate.
    Returns:
        f1 (float): The F1 score of the model on the corrupted data.
    """


    set_seed(42)  # Set seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set device to GPU if available, otherwise CPU, this is used for loading the images.
    
    drive_base_path = os.getcwd() # Get the current working directory, this is used to load the CIFAR-10-C dataset.
    # Construct the path to the CIFAR-10-C dataset
    
    cifar10_c_path = os.path.join(drive_base_path, extract_cifar10c_corruption(download_cifar10c_folder("data")))
    corruption_file = os.path.join(cifar10_c_path, f"{corruption_type}.npy") 
    
    test_set = CIFAR10(root='./data', train=False, download=True) # Load the CIFAR-10 test set, which contains 10,000 images and their corresponding labels.
    true_labels = torch.tensor(test_set.targets)  # Should have 10,000 labels
    
    
    data = np.load(corruption_file)[(severity - 1) * 10000: severity * 10000] # Load the corrupted images for the specified severity level.
    data = torch.tensor(data).permute(0, 3, 1, 2).float() / 255.0  # Normalize to [0,1]
    
    # Normalize the data using the CIFAR-10 mean and standard deviation.
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1) 
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
    data = (data - mean) / std

    dataset = TensorDataset(data, true_labels) # Create a TensorDataset with the corrupted images and their true labels.
    loader = DataLoader(dataset, batch_size=128, shuffle=False) # Create a DataLoader for the dataset, with a batch size of 128 and no shuffling.

    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculation for evaluation
        for images, labels in loader: # Iterate over the DataLoader
            images = images.to(device) # Move images to the device (GPU or CPU)
            outputs = model(images) # Forward pass through the model
            _, predicted = outputs.max(1) # Get the predicted class labels

            all_preds.extend(predicted.cpu().numpy()) # Extend the list of predictions with the predicted labels
            all_labels.extend(labels.numpy()) # Extend the list of true labels with the true labels

    f1 = f1_score(all_labels, all_preds, average='macro')  # Calculate the F1 score using macro averaging, could also use 'micro' or 'weighted' depending on the use case.
    return f1

def evaluate_model_on_all_corruptions (model):
    """
    Evaluate the model on all corruption types and severity levels from CIFAR-10-C dataset.
    Args:
        model: The model to evaluate.
    Returns:
        results (list): A list of dictionaries containing corruption type, severity level, and F1 score."""
    
    set_seed(42)  # Set seed for reproducibility
    
    # List of corruption types to evaluate
    corruptions = [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog", "brightness",
        "contrast", "elastic_transform", "pixelate", "jpeg_compression"
    ]

    results = []

    for corruption in tqdm(corruptions): # Iterate over each corruption type
        for severity in range(1, 6):
            f1 = evaluate_model_on_corruption(corruption, severity, model) # Evaluate the model on the corruption type and severity level
            results.append({
                'corruption': corruption,
                'severity': severity,
                'f1_macro': f1
            }) # Append the results to the list
        
    return results

def attack_model(model, device, batch_size=1):
    """ Run a black-box attack on the model using Foolbox's Boundary Attack.
    This function loads images from the CIFAR-10 dataset, applies the Boundary Attack, and returns the clean accuracy, robust accuracy, and perturbation size.
    The attack is performed on a random sample of images from the CIFAR-10 dataset.
    The function uses a fixed seed for reproducibility.

    Args:
        model: The model to evaluate.
        device : The device to run the model on (CPU or GPU).
        batch_size (int, optional): The number of images on which to run an attack. Defaults to 1.

    Returns:
        metrics (float): returns a tuple containing the mean and standard deviation of the clean accuracy, robust accuracy, and perturbation size over all images in the batch.
    """
    
    set_seed(42) # Set seed for reproducibility
    
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    # Load images from CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Create a random sampler for the desired batch size
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)  # Shuffle with the set seed
    sampled_indices = indices[:batch_size]
    
    clean_accuracies = []
    robust_accuracies = []
    perturbation_sizes = []
    
    for i in tqdm(sampled_indices):
        image, label = dataset[i]
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        label = torch.tensor([label]).to(device)

        # Run black-box attack
        attack = fb.attacks.BoundaryAttack()
        
        clean_accuracy = fb.accuracy(fmodel, image, label)
        
        if clean_accuracy == 0.0:
            print("Model is not accurate on the clean image. Skipping attack.")
            clean_accuracies.append(0.0)
            robust_accuracies.append(0.0) # Calculate robust accuracy which is the accuracy of the model when it is attacked
            perturbation_sizes.append(0.0)
        else :
            # Run the attack
            raw_advs, clipped_advs, success = attack(fmodel, image, label, epsilons=None)
            # Collect metrics
            clean_accuracies.append(clean_accuracy)
            robust_accuracies.append(1 - success.cpu().numpy().mean()) # Calculate robust accuracy which is the accuracy of the model when it is attacked
            perturbation_sizes.append((clipped_advs - image).norm().cpu().numpy().mean())
        
    return np.mean(clean_accuracies), np.mean(robust_accuracies), np.mean(perturbation_sizes), np.std(clean_accuracies), np.std(robust_accuracies), np.std(perturbation_sizes)


def evaluate_on_clean_testset(model):
    """
    Evaluate the model on the clean CIFAR-10 test set.
    Args:
        model: The model to evaluate.
    Returns:
        f1 (float): The F1 score of the model on the clean test set.
    """
    set_seed(42)  # Set seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device to GPU if available, otherwise CPU
    
    # Load the CIFAR-10 test set
    # Apply the same normalization as used during training
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
    ])
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize lists to store predictions and labels
    all_preds = []
    all_labels = []

    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for evaluation
        for images, labels in test_loader: # Iterate over the test DataLoader
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average='macro') # Calculate the F1 score using macro averaging
    return f1

def download_cifar10c_corruption(corruption_type, download_dir='./data/new_CIFAR-10-C'):
    """
    Downloads the specified corruption .npy file from the official CIFAR-10-C repository if not present.
    """
    os.makedirs(download_dir, exist_ok=True)
    corruption_file = os.path.join(download_dir, f"{corruption_type}.npy")
    if not os.path.exists(corruption_file):
        url = f"https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
        print(f"Downloading {corruption_type}.npy from {url} ...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(corruption_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            raise RuntimeError(f"Failed to download {corruption_type}.npy (status code: {response.status_code})")
    return corruption_file