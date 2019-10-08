from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from src.models.SuperModel import SuperModel
from src.models.NaiveBayes import NaiveBayes
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def get_vectorizer(vectorizer_name):
    if vectorizer_name == "BINARY":
        return CountVectorizer(min_df=0.0001, stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2),
                                     strip_accents='ascii', binary=True)
    elif vectorizer_name == "TFIDF":
        return TfidfVectorizer(min_df=0.0001, stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2),
                                     strip_accents='ascii')
    else:
        raise Exception("The type of vectorizer " + vectorizer_name + " is not known")

def get_model(model_name: str):
    if model_name == "LR":
        return LogisticRegression(solver='lbfgs', multi_class='auto')
    elif model_name == "NB":
        return NaiveBayes()
    elif model_name == "GNB":
        return GaussianNB()
    elif model_name == "KNN":
        return KNeighborsClassifier()
    elif model_name == "DT":
        return DecisionTreeClassifier()
    elif model_name == "RF":
        return RandomForestClassifier(n_estimators=150, max_depth=10, random_state=0, class_weight='balanced')
    elif model_name == "SVM":
        return SVC(kernel='linear', decision_function_shape='ovr', class_weight='balanced')
    elif model_name == "SUPER":
        return SuperModel()
    else:
        raise Exception("The model " + model_name + " is not recognized")
