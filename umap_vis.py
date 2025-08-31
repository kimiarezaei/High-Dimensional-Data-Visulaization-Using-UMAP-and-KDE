import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import umap
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

from utils import my_data


def plot_umap_3D(data_path, save_dir):
    """
    Perform PCA on the dataset and plot the first two principal components.

    Parameters:
    data_path (str): Path to the CSV file containing the dataset.
    """
    # Load the dataset
    df = my_data(data_path)

    # split the data into features and target
    X = df.drop(columns=['class'])  # All features
    y = df['class']  # Labels 

    count_severe = df[df['class']==1].shape[0]  # Count of class 1
    count_mild = df[df['class']==0].shape[0]  # Count of class 0
    print(f"Number of class one data: {count_severe}", 
          f"Number of class zero data: {count_mild}")

    # Standardize the features
    X_scaled = StandardScaler().fit_transform(X)  # Normalize the features

    #Apply umap to reduce to 3D
    reducer = umap.UMAP(n_components=3)  
    X_umap_3d = reducer.fit_transform(X_scaled)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot class 0
    ax.scatter(X_umap_3d[y == 0, 0], X_umap_3d[y == 0, 1], X_umap_3d[y == 0, 2],
            color='aqua', marker='*', alpha=0.7, label='class 0')

    # Plot class 1
    ax.scatter(X_umap_3d[y == 1, 0], X_umap_3d[y == 1, 1], X_umap_3d[y == 1, 2],
            color='deeppink', marker='o', alpha=0.7, label='class 1')

    # Axis labels
    # ax.set_title("3D UMAP Projection")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    ax.legend(loc='upper right')
    # Save images from multiple angles efficiently
    for i, angle in enumerate(range(0, 360, 45)):
        ax.view_init(elev=30, azim=angle)
        fig.canvas.draw()  # Force redraw with new view
        plt.savefig(f"{save_dir}/umap_3d_angle_{angle}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Animation function
    def rotate(angle):
        ax.view_init(elev=30, azim=angle)
        
    # Create animation (rotate from 0° to 360°)
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50)
    ani.save(f'{save_dir}/umap_rotation.gif', writer='pillow', fps=20)



def plot_umap_2D(data_path, save_dir):
    """
    Perform PCA on the dataset and plot the first two principal components.

    Parameters:
    data_path (str): Path to the CSV file containing the dataset.
    """
    # Load the dataset
    df = my_data(data_path)

    # split the data into features and target
    X = df.drop(columns=['class'])  # All features
    y = df['class']  # Labels 

    count_severe = df[df['class']==1].shape[0]  # Count of class 1
    count_mild = df[df['class']==0].shape[0]  # Count of class 0
    print(f"Number of class one data: {count_severe}", 
          f"Number of class zero data: {count_mild}")

    # Standardize the features
    X_scaled = StandardScaler().fit_transform(X)  # Normalize the features

    #Apply umap to reduce to 3D
    reducer = umap.UMAP(n_components=3)  
    X_umap_2d = reducer.fit_transform(X_scaled)

    # Plot in 2D
    plt.figure(figsize=(10, 7))

    # Plot class 0
    plt.scatter(X_umap_2d[y == 0, 0], X_umap_2d[y == 0, 1],
                color='aqua', marker='*', alpha=0.7, label='class 0')

    # Plot class 1
    plt.scatter(X_umap_2d[y == 1, 0], X_umap_2d[y == 1, 1],
                color='deeppink', marker='o', alpha=0.7, label='class 1')

    # Axis labels
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    # plt.title("2D UMAP Projection")
    plt.legend(loc='upper right')

    # Save figure
    plt.savefig(f'{save_dir}/umap_2d.png', dpi=300, bbox_inches='tight')
    plt.show()



