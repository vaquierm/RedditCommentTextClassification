import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels


from src.config import processed_dir_path, vocabularies_to_run, vectorizers_to_run, models_to_run, run_grid_search, results_dir_path
from src.utils.utils import get_training_feature_matrix, save_confusion_matrix, create_results_file, append_results
from src.utils.factory import get_vectorizer, get_model


# This file contains the automation of converting all the raw data to feature vectors


def run_validation_pipeline(linear_correlation: bool = False):

    results_data_file = os.path.join(results_dir_path, "results.txt")

    print("\n\nValidating models against k fold validation...")

    # For each vocabulary, load a dictionary and the corresponding raw data,
    # Convert it all to a feature vector, and save it to csv in the processed directory
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

            print("\t\tVectorized input has shape: " + str(X.shape))

            if linear_correlation:
                X = remove_low_correlation_features(X, Y)
                print("\t\tThe new vectorized input has shape: " + str(X.shape))

            for model_to_run in models_to_run:
                model = get_model(model_to_run, run_grid_search)

                if not run_grid_search or not 'GridSearch' in str(type(model)):
                    print("\t\t\tRunning k fold validation on model: " + model_to_run)
                    # For each model run kfold validation
                    Y_pred = k_fold_validation(model, X, Y)
                else:
                    print("\t\t\tRunning grid search on model: " + model_to_run)
                    # If we want to run gridsearh
                    model.fit(X, Y)

                    print("\t\t\tThe best parameters for model: " + model_to_run + " are ", model.best_params_)
                    print("\t\t\tRunning k fold validation with the best model")
                    Y_pred = k_fold_validation(model.best_estimator_, X, Y)

                # get confusion matrix and save to png
                conf_mat = confusion_matrix(Y, Y_pred)
                results_confusion_matrix_file = os.path.join(results_dir_path, vocabulary + "_"+ vec + "confusion.png")
                save_confusion_matrix(conf_mat, "Confusion Matrix for vocabulary " + vocabulary + " and vectorizer " + vec, unique_labels(Y_pred, Y), results_confusion_matrix_file)

                # get accuracy
                accuracy = accuracy_score(Y, Y_pred)
                print("\t\t\t\tAccuracy of model " + model_to_run + ": ", accuracy)
                append_results(model.__class__.__name__ + ": " + str(accuracy), results_data_file)

            # save the accuracies of vocab for each model???
            # save_accuracy_bargraph(accuracies)

        print("Validation on all models")


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


def remove_low_correlation_features(X, Y, return_index_array: bool = False):
    # Calculate the correlation of input to output
    print("\t\tCalculating F score")
    p_scores = f_classif(X, Y)[1]

    p_scores = np.log(X.getnnz(axis=0)) * p_scores
    # Get indicies of high p_score features
    high_pscores = (p_scores.mean() - 0.4 * p_scores.std()) > p_scores

    if not return_index_array:
        return X[:, high_pscores]
    else:
        return X[:, high_pscores], high_pscores


if __name__ == '__main__':
    run_validation_pipeline()
