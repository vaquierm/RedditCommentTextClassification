# This file contains all automation for validating the different models against different vocabularies


def main(processed_dir_path: str = "../../data/processed_data", vocabularies_dir_path: str = "../../data/vocabularies", results_dir_path: str = "../../results"):
    vocabularies_to_run = ["LEMMATIZED", "CORRELATED"]
    models_to_run = ["LR", "NB", "DT", "SVM", "SUPER"]
    # TODO: Implement the validation pipeline
    # Load all the specified vocabularies in Dictionaries, run each model on each dictionary.
    # Generate graphs and txt files for results in the results folder.
    # https://github.com/vaquierm/RedditCommentTextClassification/issues/6
    raise Exception("Not implemented yet")


if __name__ == "__main__":
    main()
