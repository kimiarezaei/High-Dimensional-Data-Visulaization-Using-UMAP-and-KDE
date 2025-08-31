import os
import pandas as pd


def folder_creator(dir_path):
    """create a new folder

    Args:
        dir_path (string): path to make the new folder
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


#reading data from CSV files and concatenate them
def my_data(path_list):
    df = pd.read_csv(path_list)  

    df_final = df.dropna()
    return df_final
