import os

from src.create_vocabularies import create_vocabularies
from src.validation_pipeline import run_validation_pipeline
from src.config import raw_data_dir_path,processed_dir_path,vocabularies_dir_path,results_dir_path


def main():
    # Check if all directories exist
    if not os.path.isdir(raw_data_dir_path):
        raise Exception("The raw data folder does not exist")
    if not os.path.isdir(processed_dir_path):
        raise Exception("The processed data folder does not exist")
    if not os.path.isdir(vocabularies_dir_path):
        raise Exception("The vocabularies folder does not exist")
    if not os.path.isdir(results_dir_path):
        raise Exception("The results folder does not exist")

    # First create all the specified vocabularies from the original raw data
    # Create copies of the original raw data by taking into account the lemmatization, feature selection etc
    create_vocabularies()

    # then run a k-fold validation against each one of them
    # All results produced here will be found in the results folder
    run_validation_pipeline()


if __name__ == '__main__':
    main()
