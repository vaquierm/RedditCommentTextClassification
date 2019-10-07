# This file contains all functions to related to creating a clean vocabulary
import nltk
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

# Lemmatization was compared using diff libraries https://www.machinelearningplus.com/nlp/lemmatization-examples-python/


def create_vocab(comments_train: list, comments_test: list, vocab_type: str):
    """
    Creates a vocabulary from the training set of comments.
    Both the training and testing vocabularies are processed and returned
    :param comments_train: All raw comments from the training set
    :param comments_test: All raw comments from the testing set
    :param vocab_type: Possible values are 'LEMMA' and 'STEM'
    :return: The processed list of training comments, The processed list of test comments
    """

    # list of stopwordList is larger than nltk.stopwords but does it even matter due to lemmatization
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


    # Preprocess the dataset bu applying custom replacers
    print("\t\tApplying custom replacers")
    comments_train = replace_all_for_strong_vocab(comments_train)

    # Get the root of each words
    if vocab_type == "LEMMA":
        print("\t\tLemmatizing training set")
        comments_train = lemmatize_comments(comments_train)
        print("\t\tLemmatizing test set")
        comments_test = lemmatize_comments(comments_test)
    elif vocab_type == "STEM":
        print("\t\tStemming training set")
        comments_train = stem_comments(comments_train)
        print("\t\tStemming test set")
        comments_test = stem_comments(comments_test)

    return comments_train, comments_test


def lemmatize_comments(comments):
    """
    Lemmatizes all the words in all the comments
    :param comments: List of comments to lemmatize
    :return: List of comments with all the words lemmatized
    """

    lemmatized_comments = []
    lemmatizer = WordNetLemmatizer()

    for i in range(len(comments)):
        sentence = []
        for w in nltk.word_tokenize(comments[i]):

            word = lemmatizer.lemmatize(w, get_wordnet_pos(w))
            word = reduce_lengthening(word)
            sentence.append(word)

        new_sentence = ' '.join(sentence)
        lemmatized_comments.append(new_sentence)
    return lemmatized_comments


def stem_comments(comments):
    """
    Stems all the words of all the comments
    :param comments: List of comments to stem
    :return: List of comments with all the words stemmed
    """

    stemmed_comments = []
    port_stemmer = PorterStemmer()

    for i in range(len(comments)):
        sentence = []
        for w in nltk.word_tokenize(comments[i]):
            word = port_stemmer.stem(w)
            word = reduce_lengthening(word)
            sentence.append(word)

        new_sentence = ' '.join(sentence)
        stemmed_comments.append(new_sentence)
    return stemmed_comments


def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts
    :param word: word to get the POS of
    :return: POS of the word
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def reduce_lengthening(text):
    """
    Replace repeated character sequences of length 3 or greater with sequences
    of length 2.
    """
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def replace_all_for_strong_vocab(comments):
    for i in range(len(comments)):
        comments[i] = replace_youtube_links(comments[i])
        comments[i] = replace_url(comments[i])
        comments[i] = replace_smiley(comments[i])
    return comments


def replace_youtube_links(comment):
    youtube_regex = ( r'(https?://)?(www\.)?' '(youtube|youtu|youtube-nocookie)\.(com|be|ca)/' '(watch\?.*?(?=v=)v=|embed/|v/|.+\?v=)?([^&=%\?]{11})' '(\?t=((\d+)h)?((\d{2})m)?((\d{2})s)?)?')
    return re.sub(youtube_regex, 'youtubelink  ', comment)


def replace_url(comment):
    regex = (r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$')
    return re.sub(regex, 'internetlink ', comment)


def replace_smiley(comment):

    emoticonFunny = [':)', ':-)', ":')", ':P', ':D', ":'-)"]
    sentence_untokenized = ""

    tknzr = TweetTokenizer(strip_handles=False, reduce_len=True, preserve_case=False)
    commentTokenized = tknzr.tokenize(comment)

    for index_word in range(len(commentTokenized)):
        if commentTokenized[index_word] in emoticonFunny:
            commentTokenized[index_word] = "emoticonFunny"

        sentence_untokenized = sentence_untokenized + ' ' + commentTokenized[index_word]
    comment = sentence_untokenized

    return comment
