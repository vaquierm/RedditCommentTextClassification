import os
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


from src.config import processed_dir_path, vocabularies_to_run, vectorizers_to_run, models_to_run, run_grid_search
from src.utils.utils import get_training_feature_matrix
from src.utils.factory import get_vectorizer, get_model

# This file contains the automation of converting all the raw data to feature vectors


def run_validation_pipeline(linear_correlation: bool = True):

    print("\n\nValidating models against k fold validation...")

    # For each vocabulary, load a dictionary and the corresponding raw data,
    # Convert it all to a feature vector, and save it to csv in the processed directory
    for vocabulary in vocabularies_to_run:
        print("\tValidation models for vocabulary: " + vocabulary)

        for vec in vectorizers_to_run:
            print("\t\tValidation models with vectorizer: " + vec)

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

                conf_mat = confusion_matrix(Y, Y_pred)
                print("\t\t\t\tAccuracy of model " + model_to_run + ": ", accuracy_score(Y, Y_pred))

                # TODO Chloe do the result files stuff here

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


def remove_low_correlation_features(X, Y):
    # Calculate the correlation of input to output
    print("\t\tCalculating F score")
    p_scores = f_classif(X, Y)[1]

    p_scores = np.log(X.getnnz(axis=0)) * p_scores
    # Get indicies of high p_score features
    high_pscores = (p_scores.mean() - 0.4 * p_scores.std()) > p_scores

    return X[:, high_pscores]


if __name__ == '__main__':
    run_validation_pipeline()
