import csv
import pandas as pd
import numpy as np


# This file contains any util functions needed across the project.

def csv_to_python_list(file_path: str):
    """
    Reads a csv file and puts it into a python list
    :param file_path: File path of the csv file
    :return: Python list corresponding to the csv
    """
    with open(file_path, 'rb') as f:
        reader = csv.reader(f)
        list_data = list(reader)
    return list_data


def python_list_to_csv(file_path: str, list_data: list):
    """
    Writes data from a python list to a csv file
    :param file_path: File path to write the data to
    """
    with open(file_path, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(list_data)


def load_raw_training_data(file_path: str, convert_subreddits_to_number: bool = True):
    """
    Loads the raw training data from the csv file
    :param file_path: File path of raw data
    :param convert_subreddits_to_number: If true, converts all subreddits to a number corresponding to its class
    :return: List of comments, (List of associated subreddits or numpy array of numbers corresponding to the subreddits)
    """
    df = pd.read_csv(file_path)
    return list(df['comments']), list(df['subreddits'])


def load_raw_test_data(file_path: str):
    """
    Loads the raw data test from the csv file
    :param file_path: File path of raw data
    :return: numpy array (Nx1) of id's of comments, List of comments
    """
    df = pd.read_csv(file_path)
    ids = list(df['id'])
    return np.array(ids).reshape((len(ids), 1)), list(df['comments'])
