import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


from src.config import processed_dir_path, vocabularies_to_run, vectorizers_to_run, models_to_run, results_dir_path
from src.utils.utils import get_training_feature_matrix, append_results, create_results_file
from src.utils.factory import get_vectorizer, get_model


# This file contains the automation of converting all the raw data to feature vectors


def run_validation_pipeline(mutual_info: bool = False):

    print("\n\nValidating models against k fold validation...")

    # For each vocabulary, load a dictionary and the corresponding raw data,
    # Convert it all to a feature vector, and save it to csv in the processed directory
    accuracies = []
    results_data_file = os.path.join(results_dir_path, "results.txt")
    create_results_file(results_data_file)
    for vocabulary in vocabularies_to_run:
        print("\tValidation models for vocabulary: " + vocabulary)

        for vec in vectorizers_to_run:
            print("\t\tValidation models with vectorizer: " + vec)

            # write header to results file
            append_results("Accuracies for vocabulary " + vocabulary + " and vectorizer " + vec, results_data_file)

            # Create a vectoriser
            vectorizer = get_vectorizer(vec)

            raw_train_data_path = os.path.join(processed_dir_path, vocabulary + "_train_clean.csv")
            X, Y = get_training_feature_matrix(vectorizer, raw_train_data_path)

            print("\t\t\tVectorized input has shape: " + str(X.shape))

            if mutual_info:
                X = remove_low_mutual_info_features(X, Y)

            for model_to_run in models_to_run:
                print("\t\t\tRunning k fold validation on model: " + model_to_run)
                model = get_model(model_to_run)

                # For each model run kfold validation
                Y_pred = k_fold_validation(model, X, Y)

                conf_mat = confusion_matrix(Y, Y_pred)

                print(conf_mat)
                accuracy = accuracy_score(Y, Y_pred)
                print(accuracy)
                append_results(model + ": " + accuracy, results_data_file)

        print("Validation on all models")

    # save accuracies to txt file
    df_accuracies = pd.DataFrame(accuracies, columns=['Model', 'Accuracy'])
    # make a bar graph for each vocab?


def k_fold_validation(model, X, Y, k: int = 5):
    return cross_val_predict(model, X, Y, cv=k)


def remove_low_mutual_info_features(X, Y):
    # Calculate the mutual information
    print("\t\tCalculating mutual information")
    mutual_info_scores = mutual_info_classif(X, Y)

    # Get the indicies of features highly correlated
    highly_correlated_features = (mutual_info_scores > np.median(mutual_info_scores))

    # Only use the features that are highly correlated
    return X[:, highly_correlated_features]


if __name__ == '__main__':
    run_validation_pipeline()
