import os
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


from src.config import processed_dir_path, vocabularies_to_run, vectorizers_to_run, models_to_run
from src.utils.utils import get_training_feature_matrix
from src.utils.factory import get_vectorizer, get_model

# This file contains the automation of converting all the raw data to feature vectors


def run_validation_pipeline():

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

            for model_to_run in models_to_run:
                print("\t\t\tRunning k fold validation on model: " + model_to_run)
                model = get_model(model_to_run)

                # For each model run kfold validation
                Y_pred = k_fold_validation(model, X, Y)

                conf_mat = confusion_matrix(Y, Y_pred)

                print(conf_mat)
                print(accuracy_score(Y, Y_pred))

                # TODO Chloe do the result files stuff here

        print("Validation on all models")


def k_fold_validation(model, X, Y, k: int = 5):
    return cross_val_predict(model, X, Y, cv=k)


if __name__ == '__main__':
    run_validation_pipeline()
