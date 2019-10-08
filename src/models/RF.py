from src.models.Model import Model
from sklearn.ensemble import RandomForestClassifier


class RF(Model):

    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=0, class_weight='balanced')

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
