import numpy as np
import random
import sys
import argparse
from datetime import date

from utils import folder_creator
from umap_vis import plot_umap_2D, plot_umap_3D
from kde_vis import plot_kde, plot_kde_class

today = date.today()

# paths
PYTHON = sys.executable
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', 
                    default=r'your directory/file.csv', 
                    help='Directory containing data')

parser.add_argument('--save_dir', 
                    default=rf'saved files',
                    help='Directory for saving the result')

args = parser.parse_args()

# Make a folder to save files
folder_creator(args.save_dir)

# Set random seeds
np.random.seed(42)
random.seed(42)

plot_umap_3D(args.data_dir, args.save_dir)
plot_umap_2D(args.data_dir, args.save_dir)
plot_kde_class(args.data_dir, args.save_dir)
