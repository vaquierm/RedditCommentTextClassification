import os


from src.config import vocabularies_to_run, vocabularies_dir_path, raw_data_dir_path
from src.utils.utils import load_raw_test_data, load_raw_training_data
from src.data_processing.vocabulary import create_vocab

from src.utils.utils import save_cleaned_raw_data


# This file contains the automated script to create all the specified vocabularies

def create_vocabularies():
    # Check if the original raw data files are present
    train_raw_data_path = os.path.join("..", raw_data_dir_path, "reddit_train.csv")
    test_raw_data_path = os.path.join("..", raw_data_dir_path, "reddit_test.csv")
    if not os.path.isfile(train_raw_data_path):
        raise Exception("The original raw training data cannot be found at " + train_raw_data_path)
    if not os.path.isfile(test_raw_data_path):
        raise Exception("The original raw testing data cannot be found at " + train_raw_data_path)

    print("\n\nCreating vocabularies")

    # For a vocabulary for each type specified
    for vocabulary in vocabularies_to_run:

        # Load the raw dataset
        comments_train, Y_train = load_raw_training_data(train_raw_data_path, convert_subreddits_to_number=False)
        id_test, comments_test = load_raw_test_data(test_raw_data_path)

        print("\tCreating vocabulary: " + vocabulary)

        # Create the vocabs
        comments_train_clean, comments_test_clean = create_vocab(comments_train, comments_test, vocabulary)

        # Save both cleaned comment lists to csv files
        clean_raw_train_path = os.path.join("..", raw_data_dir_path, vocabulary + "_train_raw_clean.csv")
        save_cleaned_raw_data(clean_raw_train_path, train_raw_data_path, comments_train_clean)
        clean_raw_test_path = os.path.join("..", raw_data_dir_path, vocabulary + "_test_raw_clean.csv")
        save_cleaned_raw_data(clean_raw_test_path, test_raw_data_path, comments_test_clean)


if __name__ == '__main__':
    create_vocabularies()
