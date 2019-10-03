# This file contains all abstraction dealing with the conversion from reddit comment to feature vector based on a specific vobabulary


class Dictionary:

    def __init__(self, vocabulary_file_path: str):
        """
        Creates a Dictionary based on a vocabulary
        :param vocabulary_file_path: File path of the vocabulary to base this dictionary on
        """
        raise Exception("Not implemented yet")
        # TODO: Michael implement this
        # https://github.com/vaquierm/RedditCommentTextClassification/issues/5

    def comments_to_feature_vector(self, comments: list):
        """
        Converts the list of comments to a feature matrix
        :param comments: List of string containing all comments to be converted
        :return: X (NxM) where N is the number of comments and M is the number of wods in our dictionary
        """
        raise Exception("Not implemented yet")
        # TODO: Michael implement this
        # https://github.com/vaquierm/RedditCommentTextClassification/issues/10
