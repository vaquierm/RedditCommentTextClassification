from src.models.Model import Model
from sklearn.svm import LinearSVC


class LSVC(Model):

    def __init__(self):
        self.clf = LinearSVC(random_state=0, tol=1e-5)

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
