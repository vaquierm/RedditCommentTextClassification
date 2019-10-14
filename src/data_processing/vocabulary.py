# This file contains all functions to related to creating a clean vocabulary
import nltk
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer, RegexpTokenizer

# Vocab token for youtubelink
token_youtube_link = "youtubelink"
# Vocab token for internetlink
token_internet_link = "internetlink"
# Vocab token for emoticonFunny
token_emoticon_funny = "emoticonFunny"
# Vocab token for year
token_year1900 = "year1900"
token_year2000 = "year2000"

def create_vocab(comments_train: list, comments_test: list, vocab_type: str):
    """
    Creates a vocabulary from the training set of comments.
    Both the training and testing vocabularies are processed and returned
    :param comments_train: All raw comments from the training set
    :param comments_test: All raw comments from the testing set
    :param vocab_type: Possible values are 'LEMMA' and 'STEM'
    :return: The processed list of training comments, The processed list of test comments
    """

    additional_features_train = {}
    additional_features_test = {}

    # Preprocess the dataset bu applying custom replacers
    print("\t\ttrain: Get comment length and average word length in the comment")
    comments_length_train, average_word_length_train = lenghtOfComments(comments_train)
    additional_features_train.update({'comment_length': comments_length_train})
    additional_features_train.update({'average_word_length': average_word_length_train})

    print("\t\tApplying custom replacers")
    comments_train = replace_all_for_strong_vocab(comments_train)

    print("\t\tTest: Get comment length and average word length in the comment")
    comments_length_test, average_word_length_test = lenghtOfComments(comments_test)
    additional_features_test.update({'comment_length': comments_length_test})
    additional_features_test.update({'average_word_length': average_word_length_test})

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
    else:
        raise Exception("The type of vocabulary " + vocab_type + " is not known")

    return comments_train, comments_test, additional_features_train, additional_features_test
    # print(vocab)
    # print(X.toarray())


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


def lenghtOfComments(comments):
    tokenizer = RegexpTokenizer(r'\w+')
    lengthOfComments = []
    averageLengthOfWords = []
    for i in range(len(comments)):
        tokened_sentence = tokenizer.tokenize(comments[i])
        lengthOfComments.append(len(tokened_sentence))

        sum = 0
        for k in range(len(tokened_sentence)):
            lengthOfWord = len(tokened_sentence[k])
            sum += lengthOfWord
        average = round(sum/len(tokened_sentence), 0)
        averageLengthOfWords.append(average)

    return lengthOfComments, averageLengthOfWords


def replace_all_for_strong_vocab(comments):
    for i in range(len(comments)):
        comments[i] = replace_youtube_links(comments[i])
        comments[i] = replace_url(comments[i])
        comments[i] = replace_smiley(comments[i])
        comments[i] = replace_years1900(comments[i])
        comments[i] = replace_years2000(comments[i])

    return comments


def replace_youtube_links(comment):
    youtube_regex = ( r'(https?://)?(www\.)?' '(youtube|youtu|youtube-nocookie)\.(com|be|ca)/' '(watch\?.*?(?=v=)v=|embed/|v/|.+\?v=)?([^&=%\?]{11})' '(\?t=((\d+)h)?((\d{2})m)?((\d{2})s)?)?')
    return re.sub(youtube_regex, token_youtube_link, comment)


def replace_url(comment):
    regex = (r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$')
    return re.sub(regex, token_internet_link, comment)


def replace_smiley(comment):

    emoticonFunny = [':)', ':-)', ":')", ':P', ':D', ":'-)"]
    sentence_untokenized = ""

    tknzr = TweetTokenizer(strip_handles=False, reduce_len=True, preserve_case=False)
    commentTokenized = tknzr.tokenize(comment)

    for index_word in range(len(commentTokenized)):
        if commentTokenized[index_word] in emoticonFunny:
            commentTokenized[index_word] = token_emoticon_funny

        sentence_untokenized = sentence_untokenized + ' ' + commentTokenized[index_word]
    comment = sentence_untokenized

    return comment


def replace_years1900(comment):
    regex = (r'(19)\d\d')
    return re.sub(regex, token_year1900, comment)


def replace_years2000(comment):
    regex = (r'(20)\d\d')
    return re.sub(regex, token_year2000, comment)