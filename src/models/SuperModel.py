import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


from src.models.Model import Model


class SuperModel(Model):

    def __init__(self):
        super().__init__()

        # Create all the models that will be used in the ensemble # TODO: Figure out what models are good here
        self.models = []
        self.models.append(SVC(kernel='linear', decision_function_shape='ovr', class_weight='balanced'))
        self.models.append(RandomForestClassifier(n_estimators=150, max_depth=10, random_state=0, class_weight='balanced'))
        self.models.append(GaussianNB())
        self.models.append(KNeighborsClassifier())

        self.meta_model = SVC(kernel='linear', decision_function_shape='ovr', class_weight='balanced')

    def fit(self, X, Y):
        """
        Fit the model with the training data
        :param X: Inputs
        :param Y: Label associated to inputs
        :return: None
        """
        super().fit(X, Y)

        self.M = X.shape[1]

        # Split the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

        # Fit the data to each classifier in our ensemble model
        for model in self.models:
            model.fit(X_train, Y_train)

        # Now we predict the outcomes
        predictions = self.predict_all_classifiers(X_test)

        # Train the meta classifier with the predictions
        self.meta_model.fit(predictions, Y_test)

    def predict(self, X):
        """
        Predict the labels based on the inputs
        :param X: Inputs
        :return: The predicted labels based on the training
        """
        super().predict(X)

        # Predict the outcomes with all classifiers
        predictions = self.predict_all_classifiers(X)

        # Predict with the meta classifier
        return self.meta_model.predict(predictions)

    def predict_all_classifiers(self, X):
        """
        Predict the outcome with all classifiers
        :param X: X
        :return: (NxC) All outcomes for all classifiers
        """
        N = X.shape[0]

        predictions = np.empty((N, len(self.models)))
        # Predict for each model
        for i in range(len(self.models)):
            predictions[:, i] = self.models[i].predict(X)

        return predictions

    def get_params(self, deep: bool = True):
        """
        Get parameters for this estimator.
        :param deep: If True, will return the parameters for this estimator and contained subobjects that are estimators.
        :return: Parameter names mapped to their values.
        """
        model_params = []
        for i in range(len(self.models)):
            model_params.append(self.models[i].get_params(deep))

        return {'models': model_params, 'meta_models': self.meta_model.get_params(deep)}
