# Contains the raw data downloaded from https://www.kaggle.com/c/reddit-comment-classification-comp-551/data
raw_data_dir_path: str = "../data/raw_data"
# Contains all the data in feature vector form
processed_dir_path: str = "../data/processed_data"
# Contain csv files of different vocabularies
vocabularies_dir_path: str = "../data/vocabularies"
# Path to which scripts will dump data
results_dir_path: str = "../results"

# Vocab token for youtubelink
token_youtube_link = "youtubelink  "
# Vocab token for internetlink
token_internet_link = "internetlink "
# Vocab token for emoticonFunny
token_emoticon_funny = "emoticonFunny"

# These are all the different dictionary names
vocabularies_to_run = ["LEMMA", "STEM"]

# These are all the different vectorizers to run
vectorizers_to_run = ["BINARY", "TFIDF"]

# These are all the models to run and compare performance on a k fold cross validation ("LR", "NB", "DT", "SVM", "SUPER")
models_to_run = ["LR"]
