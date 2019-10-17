import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import GridSearchCV
from src.models.SuperModel import SuperModel
from src.models.LazyNaiveBayes import LazyNaiveBayes
from src.models.NaiveBayes import NaiveBayes
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB


def get_vectorizer(vectorizer_name):
    if vectorizer_name == "BINARY":
        return CountVectorizer(min_df=int(0), stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 1),
                                     strip_accents='ascii', binary=True)
    elif vectorizer_name == "TFIDF":
        return TfidfVectorizer(min_df=int(0), stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 1),
                                     strip_accents='ascii')
    else:
        raise Exception("The type of vectorizer " + vectorizer_name + " is not known")


def get_model(model_name: str, grid_search: bool = False):
    if model_name == "LR":
        if not grid_search:
            return LogisticRegression(multi_class='auto', solver='lbfgs', C=1.623776739188721, max_iter=200)
        else:
            param_grid = {
                 'C': np.logspace(-4, 4, 20),
                 'solver': ['saga', 'lbfgs']}
            return GridSearchCV(LogisticRegression(multi_class='auto'), param_grid, cv=5)
    elif model_name == "NB":
        return NaiveBayes()
    elif model_name == "NB_SKLEARN":
        return BernoulliNB()
    elif model_name == "MNNB":
        if not grid_search:
            return MultinomialNB(alpha=0.0001)
        else:
            param_grid = {
                'alpha': np.arange(0.1, 0.5, 0.05).tolist()
            }
            return GridSearchCV(MultinomialNB(), param_grid, cv=5)
    elif model_name == "KNN":
        return KNeighborsClassifier()
    elif model_name == "DT":
        return DecisionTreeClassifier()
    elif model_name == "RF":
        return RandomForestClassifier(n_estimators=500, random_state=0, class_weight='balanced')
    elif model_name == "SVM":
        if not grid_search:
            return SVC(kernel='linear', decision_function_shape='ovr', class_weight='balanced')
        else:
            param_grid = {
                'kernel': ('linear', 'rbf'),
                'C': [1, 10]
            }
            return GridSearchCV(SVC(decision_function_shape='ovr', class_weight='balanced'), param_grid, cv=5)
    elif model_name == "SUPER":
        return SuperModel()
    elif model_name == "LAZYNB":
        return LazyNaiveBayes()
    else:
        raise Exception("The model " + model_name + " is not recognized")
