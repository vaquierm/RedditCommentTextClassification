from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS


def get_vectorizer(vectorizer_name):
    if vectorizer_name == "BINARY":
        return CountVectorizer(min_df=0.0001, stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2),
                                     strip_accents='ascii', binary=True)
    elif vectorizer_name == "TFIDF":
        return TfidfVectorizer(min_df=0.0001, stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2),
                                     strip_accents='ascii')
    else:
        raise Exception("The type of vectorizer " + vectorizer_name + " is not known")
