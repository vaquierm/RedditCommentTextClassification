import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot
from src.models.Model import Model


class NaiveBayes(Model):

    def __init__(self, alpha: float = 1):
        if alpha < 0:
            raise Exception("Alpha must be greater than zero")

        self.alpha = alpha

    def fit(self, X, Y):
        """
        Fit the model with the training data
        :param X: Inputs
        :param Y: Label associated to inputs
        :return: None
        """
        super().fit(X, Y)
        # https://github.com/vaquierm/RedditCommentTextClassification/issues/1

        subreddits = np.unique(Y)

        # fit the model
        self.parameters = {}
        total_per_class = []
        thetak = []  # parameter theta k = nb comment of class 1 / total number of comments
        alpha = 1

        # compute theta k
        # for each class
        for i in range(len(subreddits)):
            feature = subreddits[i]
            numbExamples = 0

            # loop through all the comments
            for j in range(len(Y)):
                if (Y[j] == feature):
                    numbExamples += 1

            total_per_class.append(float(numbExamples))
            thetak_i = float(numbExamples) / float(X.shape[0])
            thetak.append(thetak_i)

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(Y)

        # parameter thate of kj using sparse matrices
        # add 1 for Laplace Smoothing
        kj_numerator = safe_sparse_dot(Y.T, X) + alpha
        # kj_denominator == # of comments from that class
        total_per_class = np.array(total_per_class)

        # add 2 for Laplace Smoothing
        kj_denominator = total_per_class.reshape(-1, 1) + 2*alpha

        log_thetakj = np.log(kj_numerator) - np.log(kj_denominator)

        self.parameters.update({'parameter_k': thetak})
        self.parameters.update({'parameter_log_kj': log_thetakj})

    def predict(self, X):
        """
        Predict the labels based on the inputs
        :param X: Inputs
        :return: The predicted labels based on the training
        """
        super().predict(X)
        log_one_minus_thatakj = np.log(1 - np.exp(self.parameters["parameter_log_kj"]))
        first_summation = self.parameters["parameter_log_kj"] - log_one_minus_thatakj

        first_term = np.log(self.parameters["parameter_k"])
        second_term = safe_sparse_dot(X, first_summation.T)
        third_term = log_one_minus_thatakj.sum(axis=1)
        prediction = first_term + second_term + third_term

        return np.argmax(prediction, axis=1)