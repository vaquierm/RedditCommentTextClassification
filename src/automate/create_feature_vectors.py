import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS


from src.config import raw_data_dir_path, vocabularies_dir_path, processed_dir_path, vocabularies_to_run, vectorizers_to_run
from src.utils.utils import load_raw_training_data, load_raw_test_data, save_processed_data, python_list_to_csv

# This file contains the automation of converting all the raw data to feature vectors


def convert_all_raw_data_to_feature_vectors():

    print("\n\nConverting all raw data comments to feature vectors...")

    # For each vocabulary, load a dictionary and the corresponding raw data,
    # Convert it all to a feature vector, and save it to csv in the processed directory
    for vocabulary in vocabularies_to_run:
        print("\tConverting raw data with respect to vocabulary: " + vocabulary)

        for vec in vectorizers_to_run:
            print("\t\tConverting raw data to feature vector with vectorizer: " + vec)

            # Create a vectoriser
            if vec == "BINARY":
                vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 1), strip_accents='ascii', binary=True)
            elif vec == "TFIDF":
                vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 1), strip_accents='ascii')
            else:
                raise Exception("The type of vectorizer " + vec + " is not known")

            # Get the raw data corresponding to this dictionary
            raw_train_data_path = os.path.join("..", raw_data_dir_path, vocabulary + "_train_raw_clean.csv")
            comments, Y = load_raw_training_data(raw_train_data_path)

            # Vectorize the training data
            X = vectorizer.fit_transform(comments).toarray()

            # Write the feature vectors to the processed data
            processed_train_data_file = os.path.join("..", processed_dir_path, vocabulary + "_" + vec + "_train_processed.csv")
            save_processed_data(X, Y, processed_train_data_file)

            # Get the raw test data
            raw_test_data_path = os.path.join("..", raw_data_dir_path, vocabulary + "_test_raw_clean.csv")
            ids, comments = load_raw_test_data(raw_test_data_path)

            # Convert the data to feature vector
            X = vectorizer.transform(comments).toarray()

            # Write the feature vectors to the processed data
            processed_test_data_file = os.path.join("..", processed_dir_path, vocabulary + "_" + vec + "_test_processed.csv")
            save_processed_data(X, ids, processed_test_data_file)

        print("Done converting all raw data to feature vectors")


if __name__ == '__main__':
    convert_all_raw_data_to_feature_vectors()
