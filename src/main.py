
# Contains the raw data downloaded from https://www.kaggle.com/c/reddit-comment-classification-comp-551/data
raw_data_dir_path: str = "../../data/raw_data"
# Contains all the data in feature vector form
processed_dir_path: str = "../../data/processed_data"
# Contain csv files of different vocabularies
vocabularies_dir_path: str = "../../data/vocabularies"
# Path to which scripts will dump data
results_dir_path: str = "../../results"

# These are all the different dictionary names TODO: Marine once you produce some different vocabs, put their name here
vocabularies_to_run = ["LEMMATIZED", "CORRELATED"]

# These are all the models to run and compare performance on a k fold cross validation
models_to_run = ["LR", "NB", "DT", "SVM", "SUPER"]


def main():
    print()


if __name__ == '__main__':
    main()
