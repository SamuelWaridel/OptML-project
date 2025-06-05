# .py file for functions used in visualization

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_F1_Adam(data):
    """
    Plot the results of the Adam optimizer.

    Parameters:
    df (pd.DataFrame): DataFrame containing the results of the Adam optimizer.

    Returns:
    None
    """
    plt.plot(data.columns.drop(["learning_rate", "beta_1", "beta_2"]), data.iloc[:, 3:].T, marker='o')
    plt.title("F1 Scores for Different Learning Rates and Betas")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=45)
    plt.legend("lr:"+ data['learning_rate'].astype(str) + " " + "b_1:" + data['beta_1'].astype(str) + " " + "b_2 "+ data['beta_2'].astype(str), loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid()
    plt.show()

def reshape_data_for_sns_plot (df):
    """
    Reshape the DataFrame for seaborn plotting.

    Parameters:
    df (pd.DataFrame): DataFrame containing the results of the Adam optimizer.

    Returns:
    pd.DataFrame: Reshaped DataFrame.
    """
    reshaped_data = df.melt(id_vars=["learning_rate", "beta_1", "beta_2"], var_name="epoch", value_name="f1")
    #reshaped_data['betas'] = list(zip(reshaped_data.beta_1, reshaped_data.beta_2))
    reshaped_data.rename(columns={'learning_rate': 'lr'}, inplace=True)
    return reshaped_data

def plot_F1_Adam_sns(df, hue=None, style=None, size=None, markers = False):
    """
    Plot the results of the Adam optimizer using seaborn.

    Parameters:
    df (pd.DataFrame): DataFrame containing the results of the Adam optimizer.
    hue (str): Column name to use for hue.
    style (str): Column name to use for style.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(df, x="epoch", y="f1", hue=hue, style=style, size=size, palette="tab10", markers=markers)
    plt.title("F1 Scores for Different Learning Rates and Betas")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()
    
def plot_barplot(data, ylim =(0.75, 1.0)):
    data['Beta 1 and Beta 2'] = list(zip(data['beta_1'], data['beta_2']))
    sns.catplot(data=data[data['epoch'] == 'Test'], kind='bar', x='lr', y='f1', hue='Beta 1 and Beta 2')
    plt.title("Test F1 Scores for Different Learning Rates and Betas")
    plt.xlabel("Learning Rate")
    plt.ylabel("F1 Score")
    plt.ylim(ylim)
    plt.grid()
    plt.show()
    
def plot_corruption_barplot_per_model(all_data, model_names):
    """
    Plot a bar plot of the average F1 scores for each model over all corruptions and severities.

    Parameters:
    all_data (np.ndarray): 3D array containing the F1 scores for each model, corruption, and severity.
    model_names (list): List of model names corresponding to the first dimension of all_data.

    Returns:
        None: Displays the plot.
    """
    model_means = all_data.mean(axis=(1, 2))  # mean over corruptions and severities

    plt.figure(figsize=(10, 5))
    sns.barplot(x=model_names, y=model_means)
    plt.ylabel('Average performance')
    plt.title('Average performance per model over all corruptions/severities')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def plot_corruption_curves(all_data, model_names, corruption_list):
    """
    Plot the F1 macro scores for each model across different corruption types and severity levels.
    This function creates a grid of subplots, where each subplot corresponds to a corruption type.


    Args:
        all_data (np.ndarray): 3D array containing F1 macro scores for each model, corruption type, and severity.
        model_names (list): List of model names corresponding to the first dimension of all_data.
        corruption_list (list): List of corruption types corresponding to the second dimension of all_data.
        
    Returns:
        None: Displays the plot.
    """ 
    n_models = len(model_names)
    
    fig, axes = plt.subplots(3, 5, figsize=(18, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for c_idx, corruption in enumerate(corruption_list):
        ax = axes[c_idx]
        for m_idx, model_name in enumerate(model_names):
            ax.plot(
                range(1, 6),
                all_data[m_idx, c_idx],
                label=model_name,
                linewidth=1.5
            )
        ax.set_title(corruption, fontsize=10)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_ylim(0, 1)
        ax.grid(True)

    fig.supxlabel('Severity level', fontsize=12)
    fig.supylabel('F1 macro', fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=n_models, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
    
    
def plot_corruption_barplot_per_corruption_type (all_data, corruption_list):
    """
    Plot a bar plot of the average F1 scores for each corruption type over all models and severities.

    Parameters:
    all_data (np.ndarray): 3D array containing the F1 scores for each model, corruption, and severity.
    corruption_list (list): List of corruption types corresponding to the second dimension of all_data.

    Returns:
        None: Displays the plot.
    """
    corruption_means = all_data.mean(axis=(0, 2))  

    plt.figure(figsize=(12, 6))
    sns.barplot(x=corruption_list, y=corruption_means)
    plt.ylabel('Average F1 macro')
    plt.title('Mean performance per corruption (averaged over models & severities)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
def plot_corruption_scatterplot (all_data, model_names):
    """
    Plot a scatter plot of the F1 macro scores for each model across different corruption types and severity levels.

    Args:
        all_data (np.ndarray): 3D array containing F1 macro scores for each model, corruption type, and severity.
        model_names (list): List of model names corresponding to the first dimension of all_data.
        corruption_list (list): List of corruption types corresponding to the second dimension of all_data.
        
    Returns:
        None: Displays the plot.
    """
    clean_perfs = np.array([0.912782004, 0.929615, 0.9276285529110568, 0.9112470926, 0.923232, 0.9278731261953677, 0.8250410288, 0.867643, 0.8662728703153624])
    robust_perfs = all_data.mean(axis=(1, 2))  # performance moyenne en condition corrompue

    plt.figure(figsize=(8, 6))
    plt.scatter(clean_perfs, robust_perfs)
    for i, name in enumerate(model_names):
        plt.text(clean_perfs[i] + 0.001, robust_perfs[i], name)

    plt.xlabel('Clean accuracy')
    plt.ylabel('Robust accuracy (avg corrupted)')
    plt.title('Accuracy vs. Robustness')
    plt.grid(True)
    plt.tight_layout()
    plt.show()