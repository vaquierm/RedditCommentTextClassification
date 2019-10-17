# RedditCommentTextClassification


## Directory Structure

````
.
├── data
│   ├── processed_data
│   │   ├── LEMMA_test_clean.csv
│   │   ├── LEMMA_train_clean.csv
│   │   ├── STEM_test_clean.csv
│   │   └── STEM_train_clean.csv
│   └── raw_data
│       ├── reddit_test.csv
│       └── reddit_train.csv
|
├── results
│   ├── predictions.csv
│   ├── results.txt
│   └── STEM_BINARY_DT_confusion.png
|
└── src
    ├── main.py
    ├── config.py
    |
    ├── create_vocabularies.py
    ├── validation_pipeline.py
    ├── generate_kaggle_results.py
    |
    ├── Data_Analysis.ipynb
    |
    ├── data_processing
    │   └── vocabulary.py
    |
    ├── models
    │   ├── LazyNaiveBayes.py
    │   ├── Model.py
    │   └── NaiveBayes.py
    |
    └── utils
        ├── factory.py
        └── utils.py

````

The ````data/```` folder contains all raw and parsed csv files that are used for training the model an generate predictions for the Kaggle competition.

 <br />

- The ````data/raw_data/```` contains the raw data downloaded from Kaggle containing all reddit comments. (One training file, and one test file)
- The ````data/processed_data/```` folder contains the csv files for the processed version of the raw data with all words either stemmed or lemmatized, custom regex filters applied to further reduce the feature space.

<br />
<br />

The ````results/```` folder is the default folder where all automatic scripts will dump all their results and figures
- The ````results.txt```` file will contain a detailed result of all the accuracies of each models ran on each configuration.
- The ````predictions.csv```` file contains the predictions generated to submit to Kaggle
- The ````*.confusion.png```` files are images of confusion matrices of each model ran on all different configurations

<br />
<br />

The ````src/```` folder contains all of the .py and .ipynb files. Python files directly in this folder are all top level scripts that can be ran.
- The ````config.py```` file is probably the **most important file** where all the configurations and models to be ran are defined as well as all the file paths of the raw data and result folder
- The ````create_vocabulary.py```` file is the script that will preprocess all the initial raw data to be lemmatized and stemmed. All custom regex filters are applied as well to reduce the feature space. Once the raw data is processed, it is saved into another csv file in the ````data/preprocessed_data/```` folder.
- The ````validation_pipeline.py```` file is a script that runs all different configurations and models that are defined in the ````config.py```` file and calculates the accuracy, confusion matrices, and saves all this data in the ````results/```` folder.
- The ````generate_kaggle_results.py```` file is a script that: based on the submission configuration and model defined in the ````config.py```` file, predicts the test data and generates a ````predictions.csv```` file in the ````results/```` folder to be submitted to kaggle.
- The ````main.py```` script will run all three previously described scripts in order: create_vocabulary, validation_pipeline, generate_kaggle_results.

<br />

- The ````Data_Analysis.ipynb```` is a jupyter notebook that performs an overall data analysis of the raw data to determine what kind of words are common in which subreddits etc...

- The ````data_processing/vocabulary.py```` file contains all helper functions that clean up the initial raw data by lemmatizing it, applying custom regex filters etc...

- The ````models/```` directory contains a *from scratch* implementation of Bernoulli Naive Bayes and a Lazy implementation of naive bayes using the MultinomialNB model from sklearn

- The ````utils/factory.py```` file contains all functions to get instances of models and vectorizers based on the tring keywords defines in the ````config.py```` file.

- The ````utils/utils.py```` file contains all I/O utility functions to save, load csv files, save immages etc...
