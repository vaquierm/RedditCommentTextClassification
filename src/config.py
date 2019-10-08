# Contains the raw data downloaded from https://www.kaggle.com/c/reddit-comment-classification-comp-551/data
from models.DT import DT
from models.NaiveBayes import NB
from models.LR import LR
from models.RF import RF
from models.SVC import SVC
from models.SuperModel import SuperModel


raw_data_dir_path: str = "../data/raw_data"
# Contains all the data in feature vector form
processed_dir_path: str = "../data/processed_data"
# Contain csv files of different vocabularies
vocabularies_dir_path: str = "../data/vocabularies"
# Path to which scripts will dump data
results_dir_path: str = "../results"

# Vocab token for youtubelink
token_youtube_link = "youtubelink"
# Vocab token for internetlink
token_internet_link = "internetlink"
# Vocab token for emoticonFunny
token_emoticon_funny = "emoticonFunny"

# These are all the different dictionary names ("LEMMA", "STEM")
vocabularies_to_run = ["LEMMA"]

# These are all the different vectorizers to run ("BINARY", "TFIDF")
vectorizers_to_run = ["TFIDF"]

# These are all the models to run and compare performance on a k fold cross validation ("LR", "NB", "DT", "RF", "SVM", "RF", "SUPER")
models_to_run = [LR(), NB(), DT(), RF(), SVC(), SuperModel()]

# Config to run for kaggle
kaggle_vocab = "LEMMA"
kaggle_vectorizer = "TFIDF"
kaggle_model = "LR"
