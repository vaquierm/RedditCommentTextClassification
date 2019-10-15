from src.models.Model import Model

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_classif
import numpy as np


class LazyNaiveBayes(Model):

    def __init__(self):
        super().__init__()
        # The training data set is kept
        self.X_train = None
        self.Y_train = None

        # Model used to make predictions
        self.model = GridSearchCV(MultinomialNB(), param_grid={'alpha': np.arange(0.1, 0.5, 0.05).tolist()}, cv=4)

        # Correlation scores used to do feature selection on the go
        self.p_scores = None

    def fit(self, X, Y):
        """
        Fit the model with the training data
        :param X: Inputs
        :param Y: Label associated to inputs
        :return: None
        """
        super().fit(X, Y)

        self.X_train = X
        self.Y_train = Y

        # Get the linear correlation of each feature to the output.
        self.p_scores = f_classif(X, Y)[1]

        # Weight the scores based on the number of appearances (How much can we trust each correlation score)
        self.p_scores = self.p_scores * np.log(X.getnnz(axis=0))

    def predict(self, X):
        """
        Predict the labels based on the inputs
        :param X: Inputs
        :return: The predicted labels based on the training
        """
        super().predict(X)

        # Get the non indicies of the words that are known in the feature space
        present_words = X.getnnz(axis=0)
        present_words = present_words > 0

        X_reduced = X[:, present_words]
        X_train_reduced = self.X_train[:, present_words]

        number_of_words_known = np.sum(X_reduced.getnnz(axis=1))

        p_scores_reduced = self.p_scores[present_words]

        # Now based on the correlation scores, select the best p_score indicies to train with making sure that most comment has at least 2 features
        min_score = p_scores_reduced.min()
        max_score = p_scores_reduced.max()
        increment = (max_score - min_score) / 50
        boundary = increment * 1

        while boundary < max_score:
            indicies_to_use = p_scores_reduced < boundary

            X_double_reduced = X_reduced[:, indicies_to_use]
            # Check the percentage of features that have at least two words known
            words_found = np.sum(X_double_reduced.getnnz(axis=1))

            print(words_found / number_of_words_known)
            if words_found / number_of_words_known > 0.85:
                print(X_double_reduced.shape)
                break

            boundary += increment

        # Train the model with the selected features
        self.model.fit(X_train_reduced[:, indicies_to_use], self.Y_train)

        print(self.model.best_params_)

        # Predict with this model
        return self.model.predict(X_reduced[:, indicies_to_use])
