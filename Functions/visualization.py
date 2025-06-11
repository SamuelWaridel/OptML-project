"""This module contains functions to visualize the results of model performance under different corruptions and severities.
It includes functions to plot heatmaps, bar plots, line plots, and scatter plots to analyze the robustness of models against various corruptions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
    
def plot_corruption_heatmap(all_data, model_names, corruption_list):
    """
    Plot a bar plot of the average F1 scores for each model over all corruptions and severities.

    Parameters:
    all_data (np.ndarray): 3D array containing the F1 scores for each model, corruption, and severity.
    model_names (list): List of model names corresponding to the first dimension of all_data.
    corruption_list (list): List of corruption types.

    Returns:
        None: Displays the plot.
    """
    avg_over_severity = all_data.mean(axis=2) # average over severity levels, resulting in shape (n_models, n_corruptions)

    # Prepare the display names for models and corruptions
    display_names = [name.replace('_', ', ') for name in model_names]
    display_corruption = [cor.replace('_', ' ') for cor in corruption_list]

    df_heatmap = pd.DataFrame(
        avg_over_severity,
        index=display_names,
        columns=display_corruption
    ) # Convert to DataFrame for better visualization

    # Plotting the heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(df_heatmap, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Average F1-score over severities'})
    plt.xlabel("Type of Corruption")
    plt.ylabel("Model (Architecture and Optimizer)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("Best_models/Figures/corruption_heatmap.png", dpi=300, bbox_inches='tight') # Save the heatmap as a png file
    plt.show()
    
    
def plot_corruption_curve(all_data, model_names, n_severities=5):
    """
    Plot the F1 macro scores for each model across different corruption types and severity levels.
    This function creates a grid of subplots, where each subplot corresponds to a corruption type.


    Args:
        all_data (np.ndarray): 3D array containing F1 macro scores for each model, corruption type, and severity.
        model_names (list): List of model names corresponding.
        n_severity (int): Number of levels of severity for each corruption types. Default is 5.
        
    Returns:
        None: Displays the plot.
    """ 
    mean_f1_per_severity = all_data.mean(axis=1)  # average over the corruption axis, resulting in shape (n_models, n_severities)

    # Plotting the average F1 scores for each model across severity levels
    plt.figure(figsize=(8, 5))
    severities = np.arange(1, n_severities + 1)

    display_names = [name.replace('_', ', ') for name in model_names]

    for i, model_name in enumerate(display_names):
        plt.plot(severities, mean_f1_per_severity[i], marker='o', label=model_name)

    plt.xlabel("Severity Level")
    plt.ylabel("Average F1-score over all corruptions")
    plt.xticks(severities)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Architecture + Optimizer")
    plt.tight_layout()
    plt.savefig("Best_models/Figures/corruption_curve.png", dpi=300, bbox_inches='tight') # Save the plot as a png file
    plt.show()
    
def plot_corruption_barplot(all_data, corruption_list):
    """
    Plot a bar plot of the average F1 scores for each corruption type.
    This function calculates the average F1 score across all models and severity levels for each corruption type and displays it as a bar plot.

    Parameters:
    all_data (np.ndarray): 3D array containing the F1 scores for each model, corruption, and severity.
    corruption_list (list): List of corruption types.

    Returns:
        None: Displays the plot.
    """
    # Average over all models and severities : shape (15,)
    corruption_means = all_data.mean(axis=(0, 2))  

    display_corruption = [cor.replace('_', ' ') for cor in corruption_list]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=display_corruption, y=corruption_means)
    plt.ylabel('Average F1-score across models and severities')
    plt.xlabel('Type of Corruption')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("Best_models/Figures/corruption_barplot.png", dpi=300, bbox_inches='tight') # Save the plot as a png file
    plt.show()
    
def plot_corruption_scatterplot (all_data, model_names):
    """
    Plot a scatter plot of the F1 macro scores for each model across different corruption types and severity levels.
    The x-axis represents the clean performances, while the y-axis represents the average performance over all corruptions and severities.
    This function helps visualize if certain models sacrifice clean accuracy for robustness, or vice versa.

    Args:
        all_data (np.ndarray): 3D array containing F1 macro scores for each model, corruption type, and severity.
        model_names (list): List of model names corresponding to the first dimension of all_data.
        corruption_list (list): List of corruption types corresponding to the second dimension of all_data.
        
    Returns:
        None: Displays the plot.
    
    Note:    
    The models are assumed to be in the following order, with their corresponding clean performances:
    
    0: DenseNet_AdaGrad 0.9026944891
    1: DenseNet_Adam 0.9246578628644885
    2: DenseNet_SGD 0.9125940113141272
    3: ResNet_AdaGrad 0.910335397
    4: ResNet_Adam 0.9154097303205765
    5: ResNet_SGD 0.9193440341235497
    6: VGGLike_AdaGrad 0.8219071213
    7: VGGLike_Adam 0.8581507263642598
    8: VGG_SGD 0.8676458931227247
    
    The values for clean_perfs are manually set based on the clean performances of each model on the test set after training.
    The data is found in the "Training results" subfolder, in the "Best models" folder.
    """
    
    clean_perfs = np.array([0.9026944891, 0.9246578628644885, 0.9125940113141272, 0.910335397, 0.9154097303205765, 0.9193440341235497, 0.8219071213, 0.8581507263642598, 0.8676458931227247])
    robust_perfs = all_data.mean(axis=(1, 2))  # average the performance over all corruptions and severities, resulting in shape (n_models,)

    display_names = [name.replace('_', ', ') for name in model_names]



    plt.figure(figsize=(8, 6))
    plt.scatter(clean_perfs, robust_perfs)

    for i, name in enumerate(display_names):
        # Default offsets for text placement
        dx, dy = 0.001, 0.001
        
        # Change text offsets based for specific models to avoid overlap
        if i == 7:
            dx = -0.017
        elif i == 8:
            dy = -0.004
            dx = 0.001
        elif i == 1:
            dx = -0.018
            dy = -0.003
        elif i == 4:
            dx = -0.016
        elif i == 2:
            dx = -0.016
            dy = -0.006
        elif i == 5:
            dx = -0.014
        elif i == 0:
            dx = -0.021

        plt.text(clean_perfs[i] + dx, robust_perfs[i] + dy, name, fontsize=9)



    plt.xlabel('Clean F1-score')
    plt.ylabel('Robust F1-score\n(Average F1-score over corruptions and severities)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Best_models/Figures/corruption_scatterplot.png", dpi=300, bbox_inches='tight') # Save the plot as a png file
    plt.show()