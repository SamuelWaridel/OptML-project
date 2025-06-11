"""
run.py

This script serves as the main entry point for running model analysis in the OptML-project.
It loads the best performing models that have been trained throughout this project and compares them.

The script allows the user to either run the model evaluation on the corrupted images as well as the black box attacks, or just load the results from previous evaluations.
Running the model evaluation will take a long time, especially on CPU, so it is recommended to visualize the pre-computed results.
The results are saved in the "Results/Best_models/Corruption evaluation" folder, and the black box attack results are saved in the "Results/Best_models/BlackBoxAttack.csv" file.
The script also provides various visualizations of the results, including bar plots, curves, and scatter plots.
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from Functions.implementations import *
from Functions.visualization import *
from IPython.display import clear_output
import os


# Set random seeds for reproducibility
set_seed(42) # Set seed through custom function as done throughtout the project

best_models_dir = os.path.join(os.getcwd(), "Best_models") # Directory where the best models and the relevant data is saved

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    # Give the filename of the best performing models for each optimizer
    SGD_best_models = ["SGD_VGG_Transform_lr_0.001_momentum_0.99.pth", "SGD_ResNet_Transform_lr_0.05_momentum_0.9.pth", "SGD_DenseNet_Transform_lr_0.01_momentum_0.99.pth"]
    Adam_best_models = ["Adam_VGG_lr_0.0005_beta1_0.8_beta2_0.999.pth", 'Adam_ResNet_lr_0.001_beta1_0.9_beta2_0.999.pth', 'Adam_DenseNet_lr_0.0005_beta1_0.95_beta2_0.9999.pth']
    Adagrad_best_models = ["Adagrad_VGG_lr_0.005_wd_0.0_decay_0.0.pth", "Adagrad_ResNet_lr_0.1_wd_0.001_decay_0.0.pth", "Adagrad_DenseNet_lr_0.01_wd_0.0_decay_0.0.pth"]

    # Load the best performing models
    models_dir = os.path.join(best_models_dir, "Models") # Directory where the models are saved
    sgd_models = get_best_models(SGD_best_models, models_dir, device)
    adam_models = get_best_models(Adam_best_models, models_dir, device)
    adagrad_models = get_best_models(Adagrad_best_models, models_dir, device)
    
    set_seed(42)  # Reset seed to ensure reproducibility after loading models

    list_of_optimizer_dicts = [sgd_models, adam_models, adagrad_models] # Combine the models into a list of dictionaries for easier iteration
    
    
    print("Running model evaluation...")
    # Evaluate the models on all corruptions (takes a long time on CPU)
    print("Evaluating models on all corruption types:")
    for i in range(3): # Iterate over the three optimizers
        optimizer = ["SGD", "Adam", "Adagrad"][i]
        model_dict = list_of_optimizer_dicts[i]
        print(f"Evaluating {optimizer} models...")
        for model_name, model in model_dict.items(): # Iterate over the models for each optimizer
            results = evaluate_model_on_all_corruptions(model) # Evaluate the model on all corruptions using the evaluate_model_on_all_corruptions function
            
            df = pd.DataFrame(results) # Save the results to a CSV file
            csv_path = os.path.join(os.path.join(best_models_dir, "Corruption evaluation"), f"{optimizer}_" + model_name + '.csv')
            df.to_csv(csv_path, index=False)
        clear_output(wait=True)
        
        
    print("Model evaluation completed. Results saved in the 'Corruption evaluation' folder.")
    # Run the black box attacks on the best performing models   
    print("Starting Black Box Attacks...")
    
    # Create the directory for the black box attack results
    csv_path = os.path.join(os.path.join(best_models_dir, "BlackBoxAttack.csv"))
    columns = ["optimizer", "model", "mean_clean_accuracy", "mean_robust_accuracy","avg_perturbations", "std_clean_accuracy", "std_robust_accuracy", "std_perturbations"]
    pd.DataFrame(columns=columns).to_csv(csv_path, index=False)

    for i in range(3): # Iterate over the three optimizers
        optimizer = ["SGD", "Adam", "Adagrad"][i]
        model_dict = list_of_optimizer_dicts[i]
        for model_name, model in model_dict.items(): # Iterate over the models for each optimizer
            print(f"Running black box attack on {model_name} with {optimizer} optimizer...")            
            # Run the attack and save the results
            results = attack_model(model, device, 16) # This function runs the black box attack on the model and returns the results.
            # Save the results to a CSV file
            df = pd.DataFrame([optimizer] + [model_name] + list(results)).T
            df.to_csv(csv_path, mode='a', header=False, index=False)
            clear_output(wait=True)


else:
    print("Skipping model evaluation. Loading results...")

# Load the results from the corruption evaluation
folder_path = os.path.join(best_models_dir, "Corruption evaluation")
if not os.path.exists(folder_path): # Check if the folder exists
    print(f"Folder {folder_path} does not exist. Please run the model evaluation first.")
    exit()
    

# If the folder exists, load the results
model_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')]) # Get all CSV files in the folder
model_names = [os.path.splitext(f)[0] for f in model_files] # Extract model names from the filenames

corruption_list = sorted(list({
    row['corruption']
    for file in model_files
    for _, row in pd.read_csv(os.path.join(folder_path, file)).iterrows()
}))

corruption_to_idx = {name: idx for idx, name in enumerate(corruption_list)} # Create a mapping from corruption names to indices for easier access

n_corruptions = len(corruption_list) # Number of corruption types
n_severities = 5 # Number of severity levels (1 to 5)
n_models = len(model_files) # Number of models (number of CSV files)

# Initialize a table to hold the F1 macro scores for each model, corruption type, and severity level
all_data = np.zeros((n_models, n_corruptions, n_severities))

for i, file in enumerate(model_files): # Iterate over the model files
    df = pd.read_csv(os.path.join(folder_path, file)) # Read the CSV file into a DataFrame
    for _, row in df.iterrows(): # Iterate over each row in the DataFrame and fill the table
        c_idx = corruption_to_idx[row['corruption']]
        s_idx = int(row['severity']) - 1
        all_data[i, c_idx, s_idx] = row['f1_macro']
    
print("Results loaded successfully.")

# Plot the results        
plot_corruption_curve(all_data, model_names, n_severities)
plot_corruption_scatterplot(all_data, model_names)
plot_corruption_heatmap(all_data, model_names, corruption_list)
plot_corruption_barplot(all_data, corruption_list)

