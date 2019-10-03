from src.models.Model import Model


class SuperModel(Model):

    def fit(self, X, Y):
        """
        Fit the model with the training data
        :param X: Inputs
        :param Y: Label associated to inputs
        :return: None
        """
        super().fit(X, Y)
        # TODO: Implement this
        # https://github.com/vaquierm/RedditCommentTextClassification/issues/7

    def predict(self, X):
        """
        Predict the labels based on the inputs
        :param X: Inputs
        :return: The predicted labels based on the training
        """
        super().predict(X)
        # TODO: Implement this
