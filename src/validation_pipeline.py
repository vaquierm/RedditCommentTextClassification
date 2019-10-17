import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels


from src.utils.utils import save_confusion_matrix, create_results_file, append_results, int_to_subreddit, save_accuracy_bargraph
from src.config import processed_dir_path, vocabularies_to_run, vectorizers_to_run, models_to_run, run_grid_search, results_dir_path
from src.utils.utils import get_training_feature_matrix, get_training_feature_matrix_folds
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

        accuracies = pd.DataFrame(columns=["Model", "Vectorizer", "Accuracy"])
        for vec in vectorizers_to_run:
            print("\t\tValidation models with vectorizer: " + vec)

            # write header to results file
            append_results("Accuracies for vocabulary " + vocabulary + " and vectorizer " + vec, results_data_file)

            # Create a vectoriser
            vectorizer = get_vectorizer(vec)

            raw_train_data_path = os.path.join(processed_dir_path, vocabulary + "_train_clean.csv")
            X_trains, X_tests, Y_trains, Y_tests = get_training_feature_matrix_folds(vectorizer, raw_train_data_path)
            X, Y = get_training_feature_matrix(vectorizer, raw_train_data_path)

            print("\t\tVectorized input has shape: " + str(X.shape))

            if linear_correlation:
                X = remove_low_correlation_features(X, Y)


            for model_to_run in models_to_run:
                model = get_model(model_to_run, run_grid_search)

                if not run_grid_search or not 'GridSearch' in str(type(model)):
                    print("\t\t\tRunning k fold validation on model: " + model_to_run)
                    # For each model run kfold validation
                    accuracy, conf_mat = k_fold_validation(model, X_trains, X_tests, Y_trains, Y_tests, linear_correlation)
                else:
                    print("\t\t\tRunning grid search on model: " + model_to_run)
                    # If we want to run gridsearh
                    model.fit(X, Y)

                    print("\t\t\tThe best parameters for model: " + model_to_run + " are ", model.best_params_)
                    print("\t\t\tRunning k fold validation with the best model")
                    accuracy, conf_mat = k_fold_validation(model.best_estimator_, X_trains, X_tests, Y_trains, Y_tests, linear_correlation)

                results_confusion_matrix_file = os.path.join(results_dir_path, vocabulary + "_"+ vec + "_" + model_to_run + "_" + "confusion.png")
                save_confusion_matrix(conf_mat, "Confusion Matrix for vocabulary " + vocabulary + ", vectorizer " + vec + "and model " + model_to_run, list(map(lambda pred: int_to_subreddit[pred], unique_labels(Y))), results_confusion_matrix_file)
                print("\t\t\t\tAccuracy of model " + model_to_run + ": ", accuracy)

                append_results(model_to_run + ": " + str(accuracy), results_data_file)
                accuracies = accuracies.append(pd.DataFrame({"Model": [model_to_run], "Vectorizer": [vec], "Accuracy": [accuracy]}), ignore_index=True)

        # save the accuracies of vocab for each model
        results_model_accuracy_file = os.path.join(results_dir_path, "accuracies_" + vocabulary + ".png")
        save_accuracy_bargraph(accuracies, vocabulary, results_model_accuracy_file)

        print("Validation on all models")


def k_fold_validation(model, X_trains, X_tests, Y_trains, Y_tests, linear_correlation: bool = True):

    acc = 0
    conf_matrix = np.zeros((20, 20), dtype=int)

    for i in range(len(X_trains)):
        print("\t\t\t\tFold number ", i)

        X_train = X_trains[i]
        if linear_correlation:
            X_train, index_array = remove_low_correlation_features(X_train, Y_trains[i], True)
            print("\t\t\t\tThe new vectorized input has shape: " + str(X_trains[i].shape))

        model.fit(X_train, Y_trains[i])

        X_test = X_tests[i]
        if linear_correlation:
            X_test = X_test[:, index_array]

        Y_pred = model.predict(X_test)

        acc += accuracy_score(Y_tests[i], Y_pred) / len(X_trains)
        conf_matrix += confusion_matrix(Y_tests[i], Y_pred)

    return acc, conf_matrix


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
    print("\t\tCalculating F score to remove features from array of shape ", X.shape)
    p_scores = f_classif(X, Y)[1]

    p_scores = np.log(X.getnnz(axis=0)) * p_scores
    # Get indicies of high p_score features
    high_pscores = (p_scores.mean() + 0.5 * p_scores.std()) > p_scores

    if not return_index_array:
        return X[:, high_pscores]
    else:
        return X[:, high_pscores], high_pscores


if __name__ == '__main__':
    run_validation_pipeline()
