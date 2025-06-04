"""
run.py

This script serves as the main entry point for running model analysis in the OptML-project.
It loads the best performing models that have been trained throughout this project and compares them.

Please refer to the project README for detailed usage instructions and requirements.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score, recall_score, f1_score
from Functions.implementations import *
from Functions.visualization import *
from tqdm import tqdm
import os


# Set random seeds for reproducibility
set_seed(42) # Set seed through custom function as done throughtout the project

# Load the best performing models
Adam_best_models = ["VGG_lr_0.0005_beta1_0.9_beta2_0.98.pth", 'resnet_lr_0.001_beta1_0.8_beta2_0.999.pth', 'densenet_lr_0.001_beta1_0.8_beta2_0.9999.pth']

best_models_dir = os.path.join(os.getcwd(), os.path.join("Results","Best_models"))

adam_models = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for model_name in Adam_best_models:
    model_path = os.path.join(best_models_dir, model_name)
    if os.path.exists(model_path):
        parts = model_name.split('_')
        if 'VGG' in parts[0]:
            model = VGGLike().to(device) # for vgg models
        elif 'resnet' in parts[0]:
            model = get_resnet18_cifar().to(device)
        elif 'densenet' in parts[0]:
            model = get_densenet121().to(device) # for densenet models
        else:
            print(f"Unknown model type in {model_name}. Skipping.")
            continue
        model.load_state_dict(torch.load(model_path, map_location=device))
        adam_models[parts[0]] = model
    else:
        print(f"Model {model_name} not found in {best_models_dir}")

# Print loaded models
print("Loaded models:")
for model_name, model in adam_models.items():
    print(f"{model_name}: {model}")