import os


from src.main import raw_data_dir_path, vocabularies_dir_path, processed_dir_path, vocabularies_to_run
from src.data_processing.dictionary import Dictionary
from src.utils.utils import load_raw_training_data, load_raw_test_data, save_processed_data

# This file contains the automation of converting all the raw data to feature vectors


def convert_all_raw_data_to_feature_vectors():

    print("\n\nConverting all raw data comments to feature vectors...")

    # For each vocabulary, load a dictionary and the corresponding raw data,
    # Convert it all to a feature vector, and save it to csv in the processed directory
    for vocabulary in vocabularies_to_run:
        print("\tConverting raw data with respect to vocabulary: " + vocabulary)

        # Create a dictionary
        vocab_file = os.path.join(vocabularies_dir_path, vocabulary + "_vocab.csv")
        dictionary = Dictionary(vocab_file)

        # Get the raw data corresponding to this dictionary
        raw_train_data_path = os.path.join(raw_data_dir_path, vocabulary + "_train_raw_clean.csv")
        comments, Y = load_raw_training_data(raw_train_data_path)

        # Convert the comments to a feature vector
        X = dictionary.comments_to_feature_vector(comments)

        # Write the feature vectors to the processed data
        processed_train_data_file = os.path.join(processed_dir_path, vocabulary + "_train_processed.csv")
        save_processed_data(X, Y, processed_train_data_file)

        # Get the raw test data
        raw_test_data_path = os.path.join(raw_data_dir_path, vocabulary + "_test_raw_clean.csv")
        ids, comments = load_raw_test_data(raw_test_data_path)

        # Convert the data to feature vector
        X = dictionary.comments_to_feature_vector(comments)

        # Write the feature vectors to the processed data
        processed_test_data_file = os.path.join(processed_dir_path, vocabulary + "_test_processed.csv")
        save_processed_data(X, ids, processed_test_data_file)

    print("Done converting all raw data to feature vectors")


if __name__ == '__main__':
    convert_all_raw_data_to_feature_vectors()