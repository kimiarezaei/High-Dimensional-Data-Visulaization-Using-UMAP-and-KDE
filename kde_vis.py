import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

from utils import my_data


def plot_kde(data_path, save_dir):
    """
    Perform Kernel Density Estimation (KDE) on the dataset and plot each feature in a separate subplot within one image.
    """
    # Load the dataset
    df = my_data(data_path)
    features = df.columns
    num_features = len(features)

    # Determine subplot layout (e.g., 3 columns)
    cols = 3
    rows = (num_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()  # Flatten in case of 2D array of axes

    for i, name in enumerate(features):
        feature_scaled = StandardScaler().fit_transform(df[name].values.reshape(-1, 1))
        sns.kdeplot(feature_scaled.squeeze(), ax=axes[i], color='green', label=name)
        axes[i].set_title(name)
        axes[i].set_xlabel('Feature (normalized)')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid()

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    # Save or show the image
    plt.savefig(os.path.join(save_dir, 'kde_features.png'))
    plt.close()


def plot_kde_class(data_path, save_dir):
    """
    Plot KDEs for grade 0 vs grade 1 for each feature in separate subplots within one image.
    """
    # Load dataset
    df = my_data(data_path)

    # Remove the target column from features
    features = [col for col in df.columns if col != 'class']
    num_features = len(features)

    # Split data
    df_severe = df[df['class'] == 1]
    df_mild = df[df['class'] == 0]

    # Subplot layout
    cols = 3
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, name in enumerate(features):
        # Normalize each feature independently for each class
        feature_scaled_severe = StandardScaler().fit_transform(df_severe[name].values.reshape(-1, 1)).squeeze()
        feature_scaled_mild = StandardScaler().fit_transform(df_mild[name].values.reshape(-1, 1)).squeeze()

        # KDE plots
        sns.kdeplot(feature_scaled_severe, ax=axes[i], color='red', label='grade 1')
        sns.kdeplot(feature_scaled_mild, ax=axes[i], color='blue', label='grade 0')

        # Labels and legend
        axes[i].set_title(name)
        axes[i].set_xlabel('Feature (normalized)')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid()

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'kde_features_grade0_vs_grade1.png'))
    plt.close()

      
        
        


