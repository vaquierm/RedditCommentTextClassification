import os

from src.config import kaggle_model, kaggle_vectorizer, kaggle_vocab, processed_dir_path, results_dir_path
from src.utils.utils import get_training_feature_matrix, get_testing_feature_matrix, save_kaggle_results
from src.utils.factory import get_model, get_vectorizer


def generate_kaggle_results():

    print("\n\n Generating Kaggle submission with model: " + kaggle_model + ", vocabulary: " + kaggle_vocab + ", and vectorizer: " + kaggle_vectorizer)

    # Create the appropriate vectorizer
    vectorizer = get_vectorizer(kaggle_vectorizer)

    # Load the training feature matrix
    print("\tLoading processed vocabulary: " + kaggle_vocab)
    vocabulary_path = os.path.join(processed_dir_path, kaggle_vocab + "_train_clean.csv")
    X, Y = get_training_feature_matrix(vectorizer, vocabulary_path)

    # Train model
    print("\tTraining model: " + kaggle_model)
    model = get_model(kaggle_model)
    model.fit(X, Y)

    # Lose reference to the training data
    X = None
    Y = None

    # Load the training dataset
    print("\tLoading testing data: " + kaggle_vocab)
    test_data_file_path = os.path.join(processed_dir_path, kaggle_vocab + "_test_clean.csv")
    X = get_testing_feature_matrix(vectorizer, test_data_file_path, fit=False)

    # Fit the data to the model
    print("\tPredicting data to model: " + kaggle_model)
    Y = model.predict(X)

    # Save the predicted values to the results folder
    print("\tSaving predictions...")
    results_file_path = os.path.join(results_dir_path, "predictions.csv")
    save_kaggle_results(results_file_path, Y)


if __name__ == '__main__':
    generate_kaggle_results()