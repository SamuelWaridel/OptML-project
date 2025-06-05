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
from IPython.display import clear_output
from tqdm import tqdm
import os


# Set random seeds for reproducibility
set_seed(42) # Set seed through custom function as done throughtout the project

# Ask the user if they want to run the model analysis, or just load the results
while True:
    choice = input("Do you want to load the model and evaluate them on the corrupted images (yes/no)? (if no, only the results will be loaded and displayed)").strip().lower()
    if choice in ['yes', 'no']:
        user_choice =  choice
        break
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")

if user_choice == 'yes':
    # Run the model evaluation
    print("Loading models...")
    # Load the best performing models
    SGD_best_models = []
    Adam_best_models = ["VGG_lr_0.0005_beta1_0.9_beta2_0.98.pth", 'resnet_lr_0.001_beta1_0.8_beta2_0.999.pth', 'densenet_lr_0.001_beta1_0.8_beta2_0.9999.pth']
    Adagrad_best_models = []

    best_models_dir = os.path.join(os.getcwd(), os.path.join("Results","Best_models"))

    sgd_models = get_best_models(SGD_best_models, best_models_dir)
    adam_models = get_best_models(Adam_best_models, best_models_dir)
    adagrad_models = get_best_models(Adagrad_best_models, best_models_dir)

    list_of_optimizer_dicts = [sgd_models, adam_models, adagrad_models] # Combine the models into a list of dictionaries for easier iteration
    
    
    print("Running model evaluation...")
    # Evaluate the models on all corruptions (takes a long time on CPU)
    print("Evaluating models on all corruption types:")
    for i in range(3):
        optimizer = ["SGD", "Adam", "Adagrad"][i]
        model_dict = list_of_optimizer_dicts[i]
        print(f"Evaluating {optimizer} models...")
        for model_name, model in model_dict.items():
            results = evaluate_model_on_all_corruptions(model)
            df = pd.DataFrame(results)
            csv_path = os.path.join(os.path.join(best_models_dir, "Corruption evaluation"), f"{optimizer}_" + model_name +'.csv')
            df.to_csv(csv_path, index=False)
        clear_output(wait=True)
else:
    print("Skipping model evaluation. Loading results...")

