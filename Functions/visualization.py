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