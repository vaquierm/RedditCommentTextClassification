from src.models.Model import Model
from sklearn.naive_bayes import GaussianNB


class GNB(Model):

    def __init__(self):
        self.clf = GaussianNB()

    def fit(self, X, Y):
        """
        Fit the model with the training data
        :param X: Inputs
        :param Y: Label associated to inputs
        :return: None
        """
        self.clf.fit(X, Y)

    def predict(self, X):
        """
        Predict the labels based on the inputs
        :param X: Inputs
        :return: The predicted labels based on the training
        """
        return self.clf.predict(X)