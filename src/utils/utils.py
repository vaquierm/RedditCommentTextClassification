import csv
import pandas as pd
import numpy as np
from scipy.sparse import hstack
import os


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
    if not os.path.isfile(file_path):
        raise Exception("The file " + file_path + " from which you are trying to load your dictionary does not exist")
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        list_data = list(reader)
    return list_data


def python_list_to_csv(file_path: str, list_data: list):
    """
    Writes data from a python list to a csv file
    :param file_path: File path to write the data to
    :param list_data: The python list to be saved to csv
    """
    if not os.path.isdir(os.path.dirname(file_path)):
        raise Exception("The directory " + os.path.dirname(file_path) + " to which you want to save your vocabulary data does not exist")
    with open(file_path, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(list_data)


def load_raw_training_data(file_path: str, convert_subreddits_to_number: bool = True, get_additional_features: bool = False):
    """
    Loads the raw training data from the csv file
    :param file_path: File path of raw data
    :param convert_subreddits_to_number: If true, converts all subreddits to a number corresponding to its class
    :param get_additional_features: If true, will also return a dictionary containing all additional features as vertical numpy arrays
    :return: List of comments, (List of associated subreddits or numpy array of numbers corresponding to the subreddits)
    """
    if not os.path.isfile(file_path):
        raise Exception("The file " + file_path + " from which you are trying to load your training data does not exist")

    df = pd.read_csv(file_path)
    subreddits = list(df['subreddits'])

    # If we want to convert the subreddits to numbers
    if convert_subreddits_to_number:
        subreddits = map(lambda x: subreddit_to_int[x], subreddits)
        subreddits = np.array(list(subreddits))

    if not get_additional_features:
        return list(df['comments']), subreddits
    else:
        return list(df['comments']), subreddits, get_additional_feature_arrays(df)


def load_raw_test_data(file_path: str, get_additional_features: bool = False):
    """
    Loads the raw data test from the csv file
    :param file_path: File path of raw data
    :param get_additional_features: If true, will also return a dictionary containing all additional features as vertical numpy arrays
    :return: numpy array (Nx1) of id's of comments, List of comments
    """
    if not os.path.isfile(file_path):
        raise Exception("The file " + file_path + " from which you are trying to load your test data does not exist")

    df = pd.read_csv(file_path)
    ids = list(df['id'])

    if not get_additional_features:
        return np.array(ids).reshape((len(ids), 1)), list(df['comments'])
    else:
        return np.array(ids).reshape((len(ids), 1)), list(df['comments']), get_additional_feature_arrays(df)


def get_additional_feature_arrays(df):
    """
    Get all features that were added to the processed raw data
    :param df: Dataframe
    :return: Dictionary of 'feature_name' -> vertical np array or feature
    """
    features = list(df.columns.values)

    additional_features = {}
    for feature in features:
        if feature == 'id' or feature == 'subreddits' or feature == 'comments':
            continue
        additional_features[feature] = np.array(list(df[feature])).reshape((df.shape[0], 1))

        # If contains negatives (range between -1 and 1) normalize between 0 and 1
        if additional_features[feature].min() < 0:
            additional_features[feature] = (additional_features[feature] + 1) / 4

        # If contains features greater than 1 (ex: word length etc, normalize)
        elif additional_features[feature].max() > 1:
            additional_features[feature] = (additional_features[feature] - 1) / ((additional_features[feature].max() - 1) * 10)

    return additional_features


def save_cleaned_raw_data(file_path: str, og_file_path: str, comments: list, additional_features: dict = {}):
    """
    Saves the clean raw data after lemmatization
    :param file_path: The file path to save the new clean raw data
    :param og_file_path: The file path of the origin al file path
    :param comments: The list of clean comments
    :param additional_features: Represents the custom additional features to be saved
    """
    if not os.path.isfile(og_file_path):
        raise Exception("The file " + og_file_path + " from which you are trying to load your training data does not exist")

    df = pd.read_csv(og_file_path)

    df.loc[:, 'comments'] = pd.Series(comments)

    for feature in additional_features.keys():
        df[feature] = additional_features[feature]

    df.to_csv(file_path, mode='w', index=False)


def save_processed_data(X, Y, file_path: str):
    """
    Save the arrays X and Y into a csv file
    :param X: X
    :param Y: Y
    :param file_path: File to which to save
    """
    if not os.path.isdir(os.path.dirname(file_path)):
        raise Exception("The directory " + os.path.dirname(file_path) + " to which you want to save your processed data does not exist")
    if X.shape[0] != Y.shape[0]:
        raise Exception("The cannot save X len(" + str(X.shape[0]) + ") and Y len(" + str(Y.shape[0]) + ") as they differ in length")
    combined = np.hstack((X, Y.reshape((Y.shape[0], 1))))

    np.savetxt(file_path, combined, delimiter=',')


def save_kaggle_results(result_file_path: str, Y):
    """
    Save the Kaggle predictions to a file
    :param result_file_path: File path to save to
    :param Y: Prediction results Y
    """
    if not os.path.isdir(os.path.dirname(result_file_path)):
        raise Exception("The directory " + os.path.dirname(result_file_path) + " to which you want to save Kaggle predisctions does not exist")

    ids = np.arange(Y.shape[0])
    Y = list(map(lambda pred: int_to_subreddit[pred], Y))

    # Create a dataframe
    df = pd.DataFrame({'Id': ids, 'Category': Y})

    # Save to csv
    df.to_csv(result_file_path, mode='w', index=False)


def get_training_feature_matrix(vectorizer, raw_data_path: str):
    """
    Get the training feature matrix X and labels Y
    :param vectorizer: Vectorizer for the string data
    :param raw_data_path: data path of the raw data
    :return: X, Y
    """
    # Get the raw data corresponding to this dictionary
    comments, Y, additional_features = load_raw_training_data(raw_data_path, get_additional_features=True)

    # Vectorize the training data
    X = vectorizer.fit_transform(comments)

    feature_arrays = []
    for feature in additional_features.keys():
        feature_arrays.append(additional_features[feature])

    feature_arrays.insert(0, X)

    X = hstack(feature_arrays).tocsr()

    return X, Y


def get_training_feature_matrix_folds(vectorizer, raw_data_path: str, folds: int = 5):
    # Get the raw data corresponding to this dictionary
    comments, Y, additional_features = load_raw_training_data(raw_data_path, get_additional_features=True)

    fold_length = int(len(comments)/folds)

    X_trains = []
    Y_trains = []
    X_tests = []
    Y_tests = []

    for i in range(folds):
        print("\t\t\tVectorizing fold ", i)
        split = np.arange(len(comments))
        split = np.logical_or(split < (i * fold_length), split >= ((i+1) * fold_length))
        train_comments = comments[0:i * fold_length] + comments[(i+1) * fold_length:len(comments)]
        test_comments = comments[i * fold_length:(i+1) * fold_length]

        X_trains.append(vectorizer.fit_transform(train_comments))
        X_tests.append(vectorizer.transform(test_comments))

        Y_trains.append(Y[split])
        Y_tests.append(Y[np.logical_not(split)])

    return X_trains, X_tests, Y_trains, Y_tests


def get_testing_feature_matrix(vectorizer, raw_data_path: str, fit: bool = True):
    """
    Get the testing data matrix X
    :param vectorizer: Vectorizer for the string data
    :param raw_data_path: data path of the raw data
    :param fit: If true it will perform a fit transform, otherwise, just a fit
    :return: X
    """
    ids, comments, additional_features = load_raw_test_data(raw_data_path, get_additional_features=True)

    # Vectorize the comments ad return them
    if fit:
        X = vectorizer.fit_transform(comments)
    else:
        X = vectorizer.transform(comments)

    feature_arrays = []
    for feature in additional_features.keys():
        feature_arrays.append(additional_features[feature])

    feature_arrays.insert(0, X)

    return hstack(feature_arrays).tocsr()
