import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from src.models.Model import Model


class SuperModel(Model):

    def __init__(self):
        super().__init__()

        # Create all the models that will be used in the ensemble # TODO: Figure out what models are good here
        self.models = []
        self.models.append(SVC(kernel='linear', decision_function_shape='ovr', class_weight='balanced'))
        self.models.append(SVC(kernel='rbf', decision_function_shape='ovr', class_weight='balanced'))
        self.models.append(
            RandomForestClassifier(n_estimators=150, max_depth=10, random_state=0, class_weight='balanced'))
        self.models.append(
            RandomForestClassifier(n_estimators=150, max_depth=10, random_state=0, class_weight='balanced'))

        self.meta_model = LogisticRegression()

    def fit(self, X, Y):
        """
        Fit the model with the training data
        :param X: Inputs
        :param Y: Label associated to inputs
        :return: None
        """
        super().fit(X, Y)

        # Fit the data to each classifier in our ensemble model
        for model in self.models:
            model.fit(X, Y)

    def predict(self, X):
        """
        Predict the labels based on the inputs
        :param X: Inputs
        :return: The predicted labels based on the training
        """
        super().predict(X)

        N = X.shape[0]

        predictions = np.empty((N, len(self.models)))
        # Predict for each model
        for i in range(len(self.models)):
            predictions[:, i] = self.models[i].predict(X)

        voted_predictions = np.empty(N)
        # Collect the votes of each classifiers
        for i in range(N):
            unique, counts = np.unique(predictions[i], return_counts=True)
            voted_predictions[i] = unique[np.where(counts == counts.max())[0][0]]

        return voted_predictions
