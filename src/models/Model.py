class Model:

    def __init__(self):
        self.trained = False
        self.M = None

    def fit(self, X, Y):
        """
        Fit the model with the training data
        :param X: Inputs
        :param Y: Label associated to inputs
        :return: None
        """
        self.trained = True
        self.M = X.shape[1]

        if X.shape[0] != Y.shape[0]:
            raise Exception("The number of input samples and output samples do not match!")

    def predict(self, X):
        """
        Predict the labels based on the inputs
        :param X: Inputs
        :return: The predicted labels based on the training
        """
        if not self.trained:
            raise Exception("You must train the model before predicting!")

        if self.M != X.shape[1]:
            raise Exception("The shape of your sample does not match the shape of your training data!")

    def get_params(self, deep: bool = True):
        """
        Get parameters for this estimator.
        :param deep: If True, will return the parameters for this estimator and contained subobjects that are estimators.
        :return: Parameter names mapped to their values.
        """
        return {}
