import os


from src.config import vocabularies_to_run, vocabularies_dir_path, raw_data_dir_path
from src.utils.utils import load_raw_test_data, load_raw_training_data


# This file contains the automated script to create all the specified vocabularies


def create_vocabularies():
    # Check if the original raw data files are present
    train_raw_data_path = os.path.join(raw_data_dir_path, "reddit_train.csv")
    test_raw_data_path = os.path.join(raw_data_dir_path, "reddit_test.csv")
    if not os.path.isfile(train_raw_data_path):
        raise Exception("The original raw training data cannot be found at " + train_raw_data_path)
    if not os.path.isfile(test_raw_data_path):
        raise Exception("The original raw testing data cannot be found at " + train_raw_data_path)

    # For a vocabulary for each type specified
    for vocabulary in vocabularies_to_run:
        # Load the raw dataset
        comments_train, Y_train = load_raw_training_data(train_raw_data_path, convert_subreddits_to_number=False)
        id_test, comments_test = load_raw_test_data(test_raw_data_path)

        # Do some processing and stuff by calling helper functions in src/data/processing/vocabulary.py

    # TODO: Marine implement the function to create the vocabulary
    # We want to load the training set, go through all the words, clean then (lemmatize etc...) and save the raw vocab to the vocabularies folder.
    # https://github.com/vaquierm/RedditCommentTextClassification/issues/3
    # MAKE SURE to write all your specific helper methods for this in src/data/processing/vocabulary.py
    raise Exception("Not yet Implemented")


if __name__ == '__main__':
    create_vocabularies()
