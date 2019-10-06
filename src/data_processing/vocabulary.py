# This file contains all functions to related to creating a clean vocabulary
import os
import nltk
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from src.utils.utils import load_raw_training_data
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# look this up for noo, noooooo, nooooooooo
from nltk.tokenize import TweetTokenizer

# Lemmatization was compared using diff libraries https://www.machinelearningplus.com/nlp/lemmatization-examples-python/

def main():
    create_vocab()


def create_vocab():
    # two different lists of stopwords and nltk.corpus stopwords
    # common list of stop words taken from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/stop_words.py
    stopwordList = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all",
                       "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst",
                       "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway",
                       "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes",
                       "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides",
                       "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant",
                       "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due",
                       "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough",
                       "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few",
                       "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty",
                       "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt",
                       "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers",
                       "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc",
                       "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly",
                       "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine",
                       "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely",
                       "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor",
                       "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto",
                       "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part",
                       "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming",
                       "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six",
                       "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere",
                       "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves",
                       "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these",
                       "they", "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout",
                       "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two",
                       "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
                       "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein",
                       "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole",
                       "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your",
                       "yours", "yourself", "yourselves"]

    wordsLem = stopwords.words('english')
    # pass in all the criteria necessary
    vectorizer = CountVectorizer(stop_words=wordsLem, ngram_range=(1, 2), strip_accents='ascii')
    print(vectorizer)


    raw_data_dir_path: str = "../../data/raw_data"
    train_raw_data_path = os.path.join(raw_data_dir_path, "reddit_train.csv")
    # Load the raw dataset
    comments_train, Y_train = load_raw_training_data(train_raw_data_path, convert_subreddits_to_number=False)

    comments_test = ["https://youtu.be/6xxbBR8iSZ0?t=40m49s\n\nIf you didn't find it already.\n\nNothing out of the ordinary though, she just has eye constant eye contact.https://www.youtube.com/watch?v=L9Hlj2bawFI","The striped bats are hanging on their feet for best fishes","oranges are good! :) :') <3", "AHHH, this is soooo coooool", "The dog is running, the cat ran, and the pig runs."]
    # test comments because the list is too long.
    # comments_test = []
    # comments_test.append(comments_train[2])
    print("these are the comments")
    print(comments_test)

    print("--------find_urls----------")
    url, urls_matrix = find_urls(comments_test)
    print(url)
    print(urls_matrix)

    print("----tweet_elements----------")
    tweet_element = tweet_elements(comments_test)
    print(tweet_element)

    print("----get_new_inputs_comments----------")
    inputs = get_new_input_comments(comments_test)
    print("new inputs from lemmatization")
    print(inputs)

    #todo need to find a way to tokenize, OR merge matrices for each Lemmatization, urls, smiley, and coooool. at the end.
    #inputs = ['allo', 'bien', 'bonjour', 'https://youtube.com/ahsss.eee', 'bien']
    #cannot have inputs = [['allo', 'bien'], ['bonjour', 'https://youtube.com/ahsss.eee', 'bien']]

    X = vectorizer.fit_transform(inputs)
    vocab = vectorizer.get_feature_names()
    # add manually a word URL but needs work to be part of the matrix. (maybe simply a counter per comments)
    vocab.append("URL")
    print(vocab)
    print(X.toarray())

## end of the important code. If we want other "features/constraints" add them to the new input.

# to remove ----------------
    # # todo need to have this as the original inputs. to be able to pass it to fit_transform
    # print("doing lemmatization below: is simply for plurial..., and returns a list")
    # lemmatizer = WordNetLemmatizer()
    # lemmatized_outputs = ' '.join([lemmatizer.lemmatize(w) for w in vectorizer.get_feature_names()])
    # # lemmatized_outputs is a sentence that can be appended back into the input
    # vocab2 = lemmatized_outputs.split(" ")
    # print("the list of vocabulary with repeating words")
    # print(vocab2)
    # vocab_list = []
    # # list of vocabulary that does not have repeated words.
    # for i in range(len(vocab2)):
    #     if vocab2[i] not in vocab_list:
    #         vocab_list.append(vocab[i])
    # print("The list of vocabulary without repeating words")
    # print(vocab_list)
# ---------------------------

def tweet_elements(comments):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    for element in range(len(comments)):
        tweetValues = tknzr.tokenize(comments[element])
        print(tweetValues)
    return tweetValues

# method that returns both the list of all the urls and the matrix for if any url was present
def find_urls(comments):
    urls = []
    sizeOfMatrix = len(comments) *2
    urlMatrix = np.arange(sizeOfMatrix).reshape(len(comments), 2)

    for element in range(len(comments)):
        #initialize the matrix and the counter
        count = 0
        urlMatrix[element, 0] = element
        urlMatrix[element, 1] = 0

        # tokenize in a way to keep the whole url together.
        sentence = comments[element].split()
        for w in sentence:
            if "https://" in w:
                count += 1
                urls.append(w)
                urlMatrix[element, 1] = count

    return urls, urlMatrix


# takes in parameters the list of comments ["comment1", "comment2"]
def get_new_input_comments(comments):
    lemmatized_inputs = []
    lemmatizer = WordNetLemmatizer()
    for i in range(len(comments)):
        sentence = []
        #todo tokenize in a way that the urls are kept together... or work with the method find_urls
        #Todo if already pass as words, no need for tokenization here.
        # maybe even call the link inside here.
        for w in nltk.word_tokenize(comments[i]):
        #for w in comments[element].split():

            word = lemmatizer.lemmatize(w, get_wordnet_pos(w))
            sentence.append(word)

        new_sentence = ' '.join(sentence)
        lemmatized_inputs.append(new_sentence)
    return lemmatized_inputs


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


if __name__ == "__main__":
    main()