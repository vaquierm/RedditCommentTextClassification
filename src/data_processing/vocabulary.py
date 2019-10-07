# This file contains all functions to related to creating a clean vocabulary
import nltk
import re
from src.utils.utils import load_raw_training_data
from src.utils.utils import save_cleaned_raw_data
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

# Lemmatization was compared using diff libraries https://www.machinelearningplus.com/nlp/lemmatization-examples-python/

def main():
    create_vocab()

def create_vocab(comments_train):

    # two different lists of stopwords and nltk.corpus stopwords
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

    wordsLem = stopwords.words('english')

    # pass in all the criteria necessary
    vectorizer = CountVectorizer(stop_words=stopwordList, ngram_range=(1, 2), strip_accents='ascii')

    # test comments because the list is too long.
    #comments_train = ["lmaooooooooo :) https://youtu.be/6xxbBR8iSZ0?t=40m49s\n\n I loooooove you. If you didn't find it already. connection, connected connecting, cont. \n\nNothing out of the ordinary though, she just has eye constant eye contact. (https://www.reddit.com/r/music/wiki/halloffame)","The striped bats are hanging on their feet for best fishes","oranges are good! :) :') <3 https://www.youtube.com/watch?v=L9Hlj2bawFI", "AHHH, this is soooo coooooooooooooooool", "The dog is running, the cat ran, and the pig runs."]

    #preprocess the dataset
    comments_train = replace_all_for_strong_vocab(comments_train)
    inputs = get_new_input_comments(comments_train)
    print("inputs: ", inputs)

    save_cleaned_raw_data("../data/processed_data/processed_train.csv", "../data/raw_data/reddit_train.csv", inputs)

    X = vectorizer.fit_transform(inputs)
    vocab = vectorizer.get_feature_names()
    print(vocab)
    print(X.toarray())

    return vocab, X.toarray


# This method must be called after preprocessing the data input as it will tokenize by words.
def get_new_input_comments(comments):

    lemmatized_inputs = []
    lemmatizer = WordNetLemmatizer()
    for i in range(len(comments)):
        sentence = []
        for w in nltk.word_tokenize(comments[i]):

            word = lemmatizer.lemmatize(w, get_wordnet_pos(w))
            word = reduce_lengthening(word)
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

def replace_youtube_links(sentence):
    youtube_regex = ( r'(https?://)?(www\.)?' '(youtube|youtu|youtube-nocookie)\.(com|be|ca)/' '(watch\?.*?(?=v=)v=|embed/|v/|.+\?v=)?([^&=%\?]{11})' '(\?t=((\d+)h)?((\d{2})m)?((\d{2})s)?)?')
    return re.sub(youtube_regex, 'youtubelink  ', sentence)

def replace_url(sentence):
    regex = (r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$')
    return re.sub(regex, 'internetlink ', sentence)

def replace_smiley(sentence):

    emoticonFunny = [':)', ':-)', ":')", ':P', ':D', ":'-)"]
    str = ""

    tknzr = TweetTokenizer(strip_handles=False, reduce_len=True, preserve_case=False)
    commentTokenized = tknzr.tokenize(sentence)
    # stemming the dataset here
    commentTokenized = stemming(commentTokenized)

    for index_word in range(len(commentTokenized)):
        for i in range(len(emoticonFunny)):
            if commentTokenized[index_word] == emoticonFunny[i]:
                commentTokenized[index_word] = "emoticonFunny"

        str = str + ' ' + commentTokenized[index_word]
    sentence = str

    return sentence

# pass in a list of words and return a list of words with the words stemmed.
def stemming(commentTokenized):

    list = []
    ps = PorterStemmer()

    for word in commentTokenized:
        list.append(ps.stem(word))

    return list


if __name__ == "__main__":
    main()