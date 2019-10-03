import csv
import pandas as pd
import numpy as np


# This file contains any util functions needed across the project.

subreddit_to_int = {
    'AskReddit': 0,
    'GlobalOffensive': 1,
    'Music': 2,
    'Overwatch': 3,
    'anime': 4,
    'baseball': 5,
    'canada': 6,
    'conspiracy': 7,
    'europe': 8,
    'funny': 9,
    'gameofthrones': 10,
    'hockey': 11,
    'leagueoflegends': 12,
    'movies': 13,
    'nba': 14,
    'nfl': 15,
    'soccer': 16,
    'trees': 17,
    'worldnews': 18,
    'wow': 19,
}

int_to_subreddit = {
    0: 'AskReddit',
    1: 'GlobalOffensive',
    2: 'Music',
    3: 'Overwatch',
    4: 'anime',
    5: 'baseball',
    6: 'canada',
    7: 'conspiracy',
    8: 'europe',
    9: 'funny',
    10: 'gameofthrones',
    11: 'hockey',
    12: 'leagueoflegends',
    13: 'movies',
    14: 'nba',
    15: 'nfl',
    16: 'soccer',
    17: 'trees',
    18: 'worldnews',
    19: 'wow',
}

def csv_to_python_list(file_path: str):
    """
    Reads a csv file and puts it into a python list
    :param file_path: File path of the csv file
    :return: Python list corresponding to the csv
    """
    with open(file_path, 'r') as f:
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
    subreddits = list(df['subreddits'])

    # If we want to convert the subreddits to numbers
    if convert_subreddits_to_number:
        subreddits = map(lambda x: subreddit_to_int[x], subreddits)
        subreddits = np.array(list(subreddits))

    return list(df['comments']), subreddits


def load_raw_test_data(file_path: str):
    """
    Loads the raw data test from the csv file
    :param file_path: File path of raw data
    :return: numpy array (Nx1) of id's of comments, List of comments
    """
    df = pd.read_csv(file_path)
    ids = list(df['id'])
    return np.array(ids).reshape((len(ids), 1)), list(df['comments'])
