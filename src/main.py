import os

from src.create_vocabularies import create_vocabularies
from src.validation_pipeline import run_validation_pipeline
from src.generate_kaggle_results import generate_kaggle_results
from src.config import raw_data_dir_path,processed_dir_path, results_dir_path


def main():
    # Check if all directories exist
    if not os.path.isdir(raw_data_dir_path):
        raise Exception("The raw data folder does not exist")
    if not os.path.isdir(processed_dir_path):
        raise Exception("The processed data folder does not exist")
    if not os.path.isdir(results_dir_path):
        raise Exception("The results folder does not exist")

    # First create all the specified vocabularies from the original raw data
    # Create copies of the original raw data by taking into account the lemmatization, feature selection etc
    create_vocabularies()

    # Run a k-fold validation against each one of them
    # All results produced here will be found in the results folder
    run_validation_pipeline()

    # Run the specified best model to generate results to be submitted on Kaggle
    generate_kaggle_results()


if __name__ == '__main__':
    main()
