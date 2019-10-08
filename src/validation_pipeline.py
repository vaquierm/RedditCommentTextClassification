import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


from src.config import raw_data_dir_path, vocabularies_to_run, vectorizers_to_run, models_to_run
from src.utils.utils import load_raw_training_data

# Models
from src.models.SuperModel import SuperModel
from src.models.NaiveBayes import NaiveBayes
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# This file contains the automation of converting all the raw data to feature vectors


def run_validation_pipeline():

    print("\n\nConverting all raw data comments to feature vectors...")

    # For each vocabulary, load a dictionary and the corresponding raw data,
    # Convert it all to a feature vector, and save it to csv in the processed directory
    for vocabulary in vocabularies_to_run:
        print("\tConverting raw data with respect to vocabulary: " + vocabulary)

        for vec in vectorizers_to_run:
            print("\t\tConverting raw data to feature vector with vectorizer: " + vec)

            # Create a vectoriser
            if vec == "BINARY":
                vectorizer = CountVectorizer(min_df=0.0001, stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), strip_accents='ascii', binary=True)
            elif vec == "TFIDF":
                vectorizer = TfidfVectorizer(min_df=0.0001, stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), strip_accents='ascii')
            else:
                raise Exception("The type of vectorizer " + vec + " is not known")

            raw_train_data_path = os.path.join(raw_data_dir_path, vocabulary + "_train_raw_clean.csv")
            X, Y = get_feature_matrix(vectorizer, raw_train_data_path)

            for model_to_run in models_to_run:
                model = get_model(model_to_run)

                # For each model run kfold validation
                Y_pred = k_fold_validation(model, X, Y)

                conf_mat = confusion_matrix(Y, Y_pred)

                print(conf_mat)
                print(accuracy_score(Y, Y_pred))

                # TODO Chloe do the result files stuff here


        print("Done converting all raw data to feature vectors")


def k_fold_validation(model, X, Y, k: int = 2):
    return cross_val_predict(model, X, Y, cv=k)


def get_model(model_name: str):
    if model_name == "LR":
        return LogisticRegression(solver='lbfgs', multi_class='auto')
    elif model_name == "NB":
        return NaiveBayes()
    elif model_name == "DT":
        return DecisionTreeClassifier()
    elif model_name == "SVM":
        return SVC(kernel='linear')
    elif model_name == "SUPER":
        return SuperModel()
    else:
        raise Exception("The model " + model_name + " is not recognized")


def get_feature_matrix(vectorizer, raw_data_path: str):
    # Get the raw data corresponding to this dictionary
    comments, Y = load_raw_training_data(raw_data_path)

    # Vectorize the training data
    X = vectorizer.fit_transform(comments)

    return X, Y


if __name__ == '__main__':
    run_validation_pipeline()
