from src.utils.utils import csv_to_python_list


# This file contains all abstraction dealing with the conversion from reddit comment to feature vector based on a specific vobabulary


class Dictionary:

    def __init__(self, vocabulary_file_path: str, vocabulary_name: str):
        """
        Creates a Dictionary based on a vocabulary
        :param vocabulary_file_path: File path of the vocabulary to base this dictionary on
        :param vocabulary_name: Name of the vocabulary being loaded
        """

        self.vocabulary_name = vocabulary_name

        # Load the vocabulary from the folder
        vocabulary = csv_to_python_list(vocabulary_file_path)

        # Create the mapping from word to index
        # word: str -> index_in_feature_vector: int
        self.word_map = {}
        for i in range(len(vocabulary)):
            self.word_map[vocabulary[i]] = i

        # Save the length that the feature vectors will have
        self.M = len(self.word_map.keys())

    def comments_to_feature_vector(self, comments: list):
        """
        Converts the list of comments to a feature matrix
        :param comments: List of string containing all comments to be converted
        :return: X (NxM) where N is the number of comments and M is the number of words in our dictionary
        """
        # TODO
