import os

from src.config import vocabularies_to_run, models_to_run, raw_data_dir_path, processed_dir_path, results_dir_path
from sklearn.metrics import confusion_matrix
from utils.utils import load_raw_training_data, save_results
import numpy as np
import pandas as pd


# This file contains all automation for validating the different models against different vocabularies
def run_validation_pipeline():
    """
    Run the validation pipeline over all models and vocabs
    :return: None
    """
    # TODO: Implement the validation pipeline
    # Load all the specified vocabularies in Dictionaries, run each model on each dictionary. (The lists of things to run are imported at the top)
    # Generate graphs and txt files for results in the results folder.
    # https://github.com/vaquierm/RedditCommentTextClassification/issues/6

    # fetch true values from train data file
    raw_train_data_file = os.path.join(raw_data_dir_path, "reddit_train.csv")
    (unnecessary_comments, y_true) = load_raw_training_data(raw_train_data_file)

    for vocabulary in vocabularies_to_run:
        # get the vocab from the processed csv files
        #TODO: replace the method fetch so we fetch the processed data instead (make a util function for it as well)
        # comments need to be a numpy array
        # processed_train_data_file = os.path.join(processed_dir_path, vocabulary + "_train_processed.csv")
        comments = load_raw_training_data(raw_train_data_file)
        accuracies = []

        for model in models_to_run:
            # run model for selected vocab
            accuracy = k_fold_validation(comments, y_true, model, 5)

            # calculate performance of each model -> bar graph
            accuracies.append([model, accuracy])

        # save accuracies to txt file
        df_accuracies = pd.DataFrame(accuracies, columns=['Model', 'Accuracy'])
        # make a bar graph for each vocab?
    results_data_file = os.path.join(results_dir_path, "results.txt")
    save_results(X, Y, results_data_file)


def k_fold_validation(X, Y, model, k):
    """
    Runs k-fold validation
    :param X: input
    :param Y: output
    :param model: model to run
    :param k: int: the number of times we do k-fold
    :return: float: the average accuracy from the k-fold validation
    """
    n = len(X)
    # n = X.shape[0]
    size = int(n / k)
    results = np.empty(shape=k)
    i, j = 0, 0
    total_y_pred = np.empty(shape=Y.shape)
    while i < k:
        x_true = X[j:(j + size)]
        y_true = Y[j:(j + size)]
        x_bef = X[0:j]
        y_bef = Y[0:j]
        x_aft = X[j + size:n]
        y_aft = Y[j + size:n]
        if len(x_bef) != 0 and len(x_aft) != 0:
            x_train = np.vstack((x_bef, x_aft))
        elif len(x_bef) == 0:
            x_train = x_aft
        else:
            x_train = x_bef

        if len(y_bef) != 0 and len(y_aft) != 0:
            y_train = np.vstack((y_bef, y_aft))
        elif len(y_bef) == 0:
            y_train = y_aft
        else:
            y_train = y_bef

        model.fit(x_train, y_train)
        y_pred = model.Model.predict(x_true)
        diff = y_pred - y_true
        results[i] = 1 - float(np.count_nonzero(diff)) / y_true.shape[0]
        total_y_pred[j:(j + size)] = y_pred
        i += 1; j += size

    # confusion matrix TODO: unsure where i should put this
    confusion_matrix(Y, total_y_pred)
    return np.average(results)


if __name__ == "__main__":
    run_validation_pipeline()
