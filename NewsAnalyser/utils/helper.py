"""
This file is used to store some of the basic helper functions that is used in the main app
"""
# -------------------------------------------------------------------------------------------------------------------- #
# |                                         IMPORT RELEVANT LIBRARIES                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import streamlit as st
import nltk
import spacy
import gensim

from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
from nltk.stem import WordNetLemmatizer

# -------------------------------------------------------------------------------------------------------------------- #
# |                                             DOWNLOAD DEPENDENCIES                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
with st.spinner('Downloading WordNet Corpora...'):
    nltk.download('wordnet')
    nltk.download('vader_lexicon')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                GLOBAL VARIABLES                                                  | #
# -------------------------------------------------------------------------------------------------------------------- #
lemmatizer = WordNetLemmatizer()


# -------------------------------------------------------------------------------------------------------------------- #
# |                                                HELPER FUNCTIONS                                                  | #
# -------------------------------------------------------------------------------------------------------------------- #
def readFile(filepath, fformat):
    """
    This is a helper function to read the data

    Attributes
    ----------
    filepath:                   A path-like or file-like object
    fformat:                    The format of the file to read
    ----------
    """
    if fformat == 'CSV':
        try:
            return pd.read_csv(filepath, low_memory=False, encoding='iso-8859-1')
        except Exception as e:
            st.error(f'Error: {e}')
    elif fformat == 'XLSX':
        try:
            return pd.read_excel(filepath, engine='openpyxl')
        except Exception as e:
            st.error(f'Error: {e}')
    elif fformat == 'PKL':
        try:
            return pd.read_pickle(filepath)
        except Exception as e:
            st.error(f'Error: {e}')


def readFile_dtm(filepath, fformat):
    """
        This is a helper function to read the data

        Attributes
        ----------
        filepath:                   A path-like or file-like object
        fformat:                    The format of the file to read
        ----------
        """
    if fformat == 'CSV':
        try:
            return pd.read_csv(filepath, low_memory=False, encoding='iso-8859-1', index_col=0)
        except Exception as e:
            st.error(f'Error: {e}')
    elif fformat == 'XLSX':
        try:
            return pd.read_excel(filepath, engine='openpyxl')
        except Exception as e:
            st.error(f'Error: {e}')
    elif fformat == 'PKL':
        try:
            return pd.read_pickle(filepath)
        except Exception as e:
            st.error(f'Error: {e}')


def lemmatizeText(text):
    """
    This function iterates through the pandas dataframe and lemmatizes the words
    :param text:        text to lemmatize
    :return:            str -> lemmatized words
    """
    return [lemmatizer.lemmatize(word) for word in text]


def summarise(text, stopwords, pos_tag, nlp, sent_count):
    """
    This function summarise the text dataframe

    :param text:         DataFrame
    :param nlp:          NLP model
    :param pos_tag:      Text pos tag
    :param stopwords:    Stopwords
    :return: str
    """

    try:
        # DEFINE LISTS AND DICTS
        keyword = []
        sent_strength = {}
        data = nlp(str(text))

        # EXTRACT KEYWORDS FROM TEXT
        for token in data:
            if token.text in stopwords or token.text in punctuation:
                continue
            if token.pos_ in pos_tag:
                keyword.append(token.text)

        # COUNT THE FREQUENCY OF WORDS
        freq_word = Counter(keyword)
        max_freq = Counter(keyword).most_common(1)[0][1]
        for word in freq_word.keys():
            freq_word[word] = (freq_word[word] / max_freq)

        # CALCULATE SENTENCE SCORES
        for sent in data.sents:
            for word in sent:
                if word.text in freq_word.keys():
                    if sent in sent_strength.keys():
                        sent_strength[sent] += freq_word[word.text]
                    else:
                        sent_strength[sent] = freq_word[word.text]

        # CONCATENATE THE STRINGS IN THE LIST TO A LARGER STRING
        summarized_sentences = nlargest(sent_count, sent_strength, key=sent_strength.get)
        final_sentences = [w.text for w in summarized_sentences]
        summary = ' '.join(final_sentences)
    except Exception as e:
        return text
    else:
        return summary


def sent2word(sent):
    for sentence in sent:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def stopwordRemover(text):
    stop_words = stopwords.words('english')
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in text]