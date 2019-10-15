# Contains the raw data downloaded from https://www.kaggle.com/c/reddit-comment-classification-comp-551/data


raw_data_dir_path: str = "../data/raw_data"
# Contains all the cleaned processed raw data
processed_dir_path: str = "../data/processed_data"
# Path to which scripts will dump data
results_dir_path: str = "../results"

# These are all the different dictionary names ("LEMMA", "STEM")
vocabularies_to_run = ["STEM", "LEMMA"]

# These are all the different vectorizers to run ("BINARY", "TFIDF")
vectorizers_to_run = ["TFIDF", "BINARY"]

# These are all the models to run and compare performance on a k fold cross validation ("LR", "NB", "MNNB", "KNN", "DT", "RF", "SVM", "SUPER")
models_to_run = ["LAZYNB", "MNNB"]

# If this is true, run gridsearch on each model (This will significantly increase the runtime of the validation pipeline for model types that support gridsearch)
run_grid_search = True

# Config to run for kaggle
kaggle_vocab = "STEM"
kaggle_vectorizer = "TFIDF"
kaggle_model = "LAZYNB"
