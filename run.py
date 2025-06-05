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

best_models_dir = os.path.join(os.getcwd(), os.path.join("Results","Best_models"))

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
    SGD_best_models = ["SGD_VGG_Transform_lr_0.001_momentum_0.99.pth", "SGD_ResNet_Transform_lr_0.05_momentum_0.9.pth", "SGD_DenseNet_Transform_lr_0.01_momentum_0.99.pth"]
    Adam_best_models = ["ADAM_VGG_lr_0.0005_beta1_0.9_beta2_0.98.pth", 'ADAM_ResNet_lr_0.001_beta1_0.8_beta2_0.999.pth', 'ADAM_DenseNet_lr_0.001_beta1_0.8_beta2_0.9999.pth']
    Adagrad_best_models = []

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
            csv_path = os.path.join(os.path.join(best_models_dir, "Corruption evaluation"), model_name + f"_{optimizer}" +  +'.csv')
            df.to_csv(csv_path, index=False)
        clear_output(wait=True)
else:
    print("Skipping model evaluation. Loading results...")

# Load the results from the corruption evaluation
folder_path = os.path.join(best_models_dir, "Corruption evaluation")
if not os.path.exists(folder_path):
    print(f"Folder {folder_path} does not exist. Please run the model evaluation first.")
    exit()
model_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
model_names = [os.path.splitext(f)[0] for f in model_files]

corruption_list = sorted(list({
    row['corruption']
    for file in model_files
    for _, row in pd.read_csv(os.path.join(folder_path, file)).iterrows()
}))

corruption_to_idx = {name: idx for idx, name in enumerate(corruption_list)}

n_corruptions = len(corruption_list)
n_severities = 5
n_models = len(model_files)

# Initialisation du tableau (n_models, n_corruptions, n_severities)
all_data = np.zeros((n_models, n_corruptions, n_severities))

for i, file in enumerate(model_files):
    df = pd.read_csv(os.path.join(folder_path, file))
    for _, row in df.iterrows():
        c_idx = corruption_to_idx[row['corruption']]
        s_idx = int(row['severity']) - 1
        all_data[i, c_idx, s_idx] = row['f1_macro']
    
print("Results loaded successfully.")

# Plot the results        
plot_corruption_barplot_per_model(all_data, model_names)
plot_corruption_curves(all_data, model_names, corruption_list)
plot_corruption_barplot_per_corruption_type(all_data, corruption_list)
plot_corruption_scatterplot(all_data, model_names)

