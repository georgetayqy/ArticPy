"""
This file is used to store some of the basic helper functions that is used in the main app
"""
# -------------------------------------------------------------------------------------------------------------------- #
# |                                         IMPORT RELEVANT LIBRARIES                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas
import pandas as pd
import streamlit as st
import nltk
import os

from string import punctuation
from collections import Counter
from heapq import nlargest
from nltk.stem import WordNetLemmatizer
from streamlit_pandas_profiling import st_profile_report

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


def modelIterator(model, vectoriser, top_n, vb=True):
    frame_list = []

    for id_, topic in enumerate(model.components_):
        lister = [(vectoriser.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]]
        df = pd.DataFrame(data=lister,
                          index=range(len(lister)),
                          columns=['word', 'weight'])

        if vb:
            st.markdown(f'### Topic {id_}')
            st.dataframe(df)

        frame_list.append(df)

    return frame_list


def downloadCorpora(model: str):
    """
    This function allows users to quickly and iteratively download corpora for nltk processing

    Parameter
    ----------
    models:          A list of names of models the user is trying to download
    ----------
    """
    usr_dir = os.path.expanduser(os.getcwd())
    if not isinstance(model, str):
        st.error('Error: Parameter passed in is not of type "str".')
    else:
        new_path = os.path.join(usr_dir, f'nltk_data/corpora/{model}')

        if os.path.exists(new_path):
            if not os.listdir(new_path):
                try:
                    nltk.download(model)
                except Exception as ex:
                    st.error(f'Error: {ex}')
            else:
                st.info('Corpora downloaded')
        else:
            try:
                nltk.download(model)
            except Exception as ex:
                st.error(f'Error: {ex}')


def printDataFrame(data: pandas.DataFrame, verbose_level: int, advanced: bool,
                   extract_from: object = None):
    """
    Takes in a Pandas DataFrame and prints out the DataFrame

    Parameter
    ----------
    data:                            Pandas DataFrame or Series object
    extract_from:                    Name of column to extract data from
    verbose_level:                   The number of rows of data to display
    advanced:                        Conduct Advanced Analysis on the DataFrame
    dtm:                             Special processing for DTMs
    ----------
    """

    if verbose_level != 0:
        try:
            st.dataframe(data.head(verbose_level), height=600, width=800)
        except RuntimeError:
            st.warning('Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
            st.dataframe(data.head(10), height=600, width=800)
        except KeyError:
            st.error(f'Error: DataFrame Column with value {extract_from} does not exist. Try again.')
        except Exception as ex:
            st.error(f'Error: {ex}')
        else:
            if advanced:
                if extract_from is not None:
                    with st.expander('Advanced Profile Report'):
                        st_profile_report(data[[extract_from]].profile_report(
                            explorative=True,
                            minimal=True))
                else:
                    with st.expander('Advanced Profile Report'):
                        st_profile_report(data.profile_report(
                            explorative=True,
                            minimal=True))
    else:
        try:
            st.dataframe(data, height=600, width=800)
        except RuntimeError:
            st.warning('Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
            st.dataframe(data.head(10), height=600, width=800)
        except KeyError:
            st.error(f'Error: DataFrame Column with value {extract_from} does not exist. Try again.')
        except Exception as ex:
            st.error(f'Error: {ex}')
        else:
            if advanced:
                if extract_from is not None:
                    with st.expander('Advanced Profile Report'):
                        st_profile_report(data[[extract_from]].profile_report(
                            explorative=True,
                            minimal=True))
                else:
                    with st.expander('Advanced Profile Report'):
                        st_profile_report(data.profile_report(
                            explorative=True,
                            minimal=True))


def dominantTopic(vect, model, n_words):
    """
    Returns the topic text
    """
    kw = np.array(vect.get_feature_names())
    topic_kw = []
    for weights in model.components_:
        top_kw = (-weights).argsort()[:n_words]
        topic_kw.append(kw.take(top_kw))

    return topic_kw


def mapTopic(row, dict_):
    return dict_[row]
