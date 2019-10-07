from src.models.Model import Model
from sklearn.linear_model import LogisticRegression


class LR(Model):

    def __init__(self):
        # TODO: finalise the values for Logistic Regression
        self.clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

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
