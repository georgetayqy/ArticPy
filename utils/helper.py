"""
This file is used to store some of the basic helper functions that are used frequently in the app
"""

# -------------------------------------------------------------------------------------------------------------------- #
# |                                         IMPORT RELEVANT LIBRARIES                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
import io
import logging
import os
import typing
import nltk
import numpy as np
import pandas
import pandas as pd
import plotly.utils
import streamlit as st
import pandas_profiling
import base64
import json
import pickle
import uuid
import re
import urllib.parse

from collections import Counter
from heapq import nlargest
from string import punctuation
import pymongo
from pymongo import MongoClient
from PIL import Image
from nltk.stem import WordNetLemmatizer
from streamlit_pandas_profiling import st_profile_report
from config import toolkit
from sshtunnel import SSHTunnelForwarder
from ssh_pymongo import MongoSession

# -------------------------------------------------------------------------------------------------------------------- #
# |                                             DOWNLOAD DEPENDENCIES                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
with st.spinner('Downloading WordNet Corpora...'):
    nltk.download('wordnet')
    nltk.download('vader_lexicon')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                         GLOBAL VARIABLES DECLARATION                                             | #
# -------------------------------------------------------------------------------------------------------------------- #
lemmatizer = WordNetLemmatizer()


# -------------------------------------------------------------------------------------------------------------------- #
# |                                               HELPER FUNCTIONS                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
def readFile(filepath, fformat):
    """
    This is a helper function to read the data

    Parameters
    ----------
    filepath:                           A path-like or file-like object
    fformat:                            The format of the file to read
    ----------
    """
    if fformat == 'CSV':
        try:
            return pd.read_csv(filepath, low_memory=False, encoding='latin1')
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
    elif fformat == 'JSON':
        try:
            return pd.read_json(filepath)
        except Exception as e:
            st.error(f'Error: {e}')
    elif fformat == 'HDF5':
        try:
            return pd.read_hdf(filepath)
        except Exception as e:
            st.error(f'Error: {e}')


def lemmatizeText(text):
    """
    This function iterates through the pandas dataframe and lemmatizes the words

    Parameters
    ----------
    :param text:                        Text to lemmatize (string)
    ----------
    """
    return [lemmatizer.lemmatize(word) for word in text]


def summarise(text, stopwords, pos_tag, nlp, sent_count):
    """
    This function summarise the text dataframe

    Parameters
    ----------
    text:                               DataFrame
    nlp:                                NLP model
    pos_tag:                            Text pos tag
    stopwords:                          Stopwords
    sent_count:                         Number of sentences to summarise to
    ----------
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
    except Exception:
        return text
    else:
        return summary


def modelIterator(model, vectoriser, top_n, vb=True):
    """
    This function prints out and returns the extracted topics for the NLP model passed on to it

    Parameters
    ----------
    model:                              NLP Model
    vectoriser:                         Vectorised text
    top_n:                              Number of Topics to return
    vb:                                 Verbose tag (will print out the topics if set to True
    ---------
    """
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
    models:                             A list of names of models the user is trying to download
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
                   extract_from: str or None = None):
    """
    Takes in a Pandas DataFrame and prints out the DataFrame

    Parameter
    ----------
    data:                               Pandas DataFrame or Series object
    extract_from:                       Name of column to extract data from
    verbose_level:                      The number of rows of data to display
    advanced:                           Conduct Advanced Analysis on the DataFrame
    dtm:                                Special processing for DTMs
    ----------
    """

    if verbose_level != 0:
        try:
            if extract_from is not None:
                st.dataframe(data[[extract_from]].head(verbose_level), height=600, width=800)
            else:
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
            if extract_from is not None:
                st.dataframe(data[[extract_from]], height=600, width=800)
            else:
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

    Parameters
    ----------
    vect:                               Vectorizer used
    model:                              NLP Model
    n_words:                            Number of Topics to return
    ----------
    """
    kw = np.array(vect.get_feature_names())
    topic_kw = []
    for weights in model.components_:
        top_kw = (-weights).argsort()[:n_words]
        topic_kw.append(kw.take(top_kw))

    return topic_kw


def prettyDownload(object_to_download: typing.Any, download_filename: str, button_text: str,
                   override_index: bool, format_: typing.Optional[str] = None) -> str:
    """
    Taken from Gist: https://gist.github.com/chad-m/6be98ed6cf1c4f17d09b7f6e5ca2978f

    Generates a link to download the given object_to_download.

    :rtype: object
    :param object_to_download:  The object to be downloaded, enter in a list to convert all dataframes into a final
                                Excel sheet to output
    :param download_filename: filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    :param button_text: Text to display on download button (e.g. 'click here to download file')
    :param override_index: Overrides
    :param format_: Format of the output file
    :return: the anchor tag to download object_to_download
    """

    try:
        if isinstance(object_to_download, bytes):
            pass
        elif isinstance(object_to_download, list):
            out = io.BytesIO()
            writer = pd.ExcelWriter(out, engine='openpyxl')

            for dataframe in enumerate(object_to_download):
                dataframe[1].to_excel(writer, sheet_name=f'Sheet{dataframe[0]}', index=override_index)

            writer.save()
            object_to_download = out.getvalue()
        elif isinstance(object_to_download, str):
            object_to_download = bytes(object_to_download, 'utf-8')
        elif isinstance(object_to_download, pd.DataFrame):
            if format_.lower() == 'xlsx':
                out = io.BytesIO()
                writer = pd.ExcelWriter(out, engine='openpyxl')
                object_to_download.to_excel(writer, sheet_name='Sheet1', index=override_index)
                writer.save()
                object_to_download = out.getvalue()
            elif format_.lower() == 'csv':
                object_to_download = object_to_download.to_csv(index=override_index).encode('utf-8')
            elif format_.lower() == 'json':
                object_to_download = object_to_download.to_json(index=override_index)
            elif format_.lower() == 'hdf5':
                st.warning('Output to HDF5 format is not currently supported. Defaulting to CSV instead...')
                object_to_download = object_to_download.to_csv(index=override_index).encode('utf-8')
            elif format_.lower() == 'pkl':
                out = io.BytesIO()
                object_to_download.to_pickle(path=out)
                object_to_download = out.getvalue()
            else:
                raise ValueError('Error: Unrecognised File Format')
        elif isinstance(object_to_download, plotly.graph_objs.Figure):
            object_to_download = plotly.io.to_image(object_to_download)
        elif isinstance(object_to_download, Image):
            buffer = io.BytesIO()
            object_to_download.save(buffer, format='png')
            object_to_download = buffer.getvalue()
        else:
            object_to_download = json.dumps(object_to_download)
    except Exception as ex:
        st.error(ex)
    else:
        try:
            b64 = base64.b64encode(object_to_download.encode()).decode()
        except AttributeError as e:
            b64 = base64.b64encode(object_to_download).decode()

        button_uuid = str(uuid.uuid4()).replace('-', '')
        button_id = re.sub('\d+', '', button_uuid)

        custom_css = f""" 
            <style>
                #{button_id} {{
                    background-color: rgb(255, 255, 255);
                    color: rgb(38, 39, 48);
                    padding: 0.25em 0.38em;
                    position: relative;
                    text-decoration: none;
                    border-radius: 4px;
                    border-width: 1px;
                    border-style: solid;
                    border-color: rgb(230, 234, 241);
                    border-image: initial;
                }} 
                #{button_id}:hover {{
                    border-color: rgb(246, 51, 102);
                    color: rgb(246, 51, 102);
                }}
                #{button_id}:active {{
                    box-shadow: none;
                    background-color: rgb(246, 51, 102);
                    color: white;
                    }}
            </style> """

        dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" ' \
                               f'href="data:file/txt;base64,{b64}"> {button_text}</a><br></br>'

        return dl_link


class MongoDB:
    """
    This class manages the pulling and pushing of data to and from a configured MongoDB server/MongoDB Atlas
    """

    def __init__(self, host: str = 'localhost', port: int = 27017, sshUser: str = 'user', sshPassword: str = 'password',
                 mongoUsername: str = 'user', mongoPassword: str = 'password', authDB: str = 'admin', db: str = None,
                 collection: str = None, ssh_path: os.PathLike = None):
        """
        This instantiates the MongoDB class and prepares the MongoDB connection

        :param host:                        An IP address or website where MongoDB is hosted on
        :param port:                        The port to connect to the IP address to
        :param sshUser:                     SSH Username
        :param sshPassword:                 SSH Password
        :param mongoUsername:               Username of DB User
        :param mongoPassword:               Password of DB User
        :param authDB:                      DB used for authentication
        :param db:                          Database to access
        :param collection:                  Collection to access within Database
        """

        if isinstance(host, str):
            self.host = host
        else:
            raise ValueError('Error: Invalid Host String. Only string values are accepted')

        if isinstance(port, int) and port in range(0, 65536):
            self.port = port
        else:
            raise ValueError('Error: Invalid Port')

        if isinstance(sshUser, str):
            self.sshUser = sshUser
        else:
            raise ValueError('Error: SSH Username Invalid')

        if isinstance(sshPassword, str):
            self.sshPassword = sshPassword
        else:
            raise ValueError('Error: SSH Password Invalid')

        if isinstance(mongoUsername, str):
            self.mongoUsername = mongoUsername
        else:
            raise ValueError('Error: MongoDB Username Invalid')

        if isinstance(mongoPassword, str):
            self.mongoPassword = mongoPassword
        else:
            raise ValueError('Error: MongoDB Password Invalid')

        if isinstance(authDB, str):
            self.authDB = authDB
        else:
            raise ValueError('Error: Invalid Authentication Database')

        if isinstance(db, str):
            self.db = db
        else:
            raise ValueError('Error: Invalid Database Name')

        if isinstance(collection, str):
            self.collection = collection
        else:
            raise ValueError('Error: Invalid Collection Name')

        self.ssh_path = ssh_path

        # define other constants
        self.authenticatedUser = None
        self.pointerDB = None
        self.pointerCollection = None
        self.query = None
        self.tunnel = None

    def __repr__(self):
        return repr(f'User {self.username} is connected to {self.host} through {self.port}, with the pointer at '
                    f'{self.db}.{self.collection}!')

    def __str__(self):
        return repr(f'User {self.username} is connected to {self.host} through {self.port}, with the pointer at '
                    f'{self.db}.{self.collection}!')

    def authenticate(self):
        """
        Authenticates the user to the MongoDB server
        """

        # self.uri = f'mongodb://{self.username}:{self.password}@{self.host}:{self.port}'

        try:
            self.authenticatedUser = MongoSession(
                self.host,
                port=22,
                user=self.sshUser,
                password=self.sshPassword,
                key=self.ssh_path,
                uri=f'mongodb://{self.mongoUsername}@{self.mongoPassword}@10.0.0.4:{self.port}'
            )
        except pymongo.errors.ConnectionFailure:
            st.error('Error: Connection failed to establish. Try again.')
        except pymongo.errors.ConfigurationError:
            st.error('Error: The Connection is not configured properly. Try again.')
        except pymongo.errors.PyMongoError:
            st.error('Error: Unknown fatal error. Execution failure point: Connecting to the MongoDB server.')
        except Exception as ex:
            st.error(f'Unknown Fatal Error: {ex}')
        else:
            st.success('Authenticated!')

    def point(self):
        """
        Instantiates the pointer to the database and collection of interest
        """

        try:
            self.pointerDB = self.authenticatedUser.connection[self.db]
            self.pointerCollection = self.pointerDB[self.collection]
        except pymongo.errors.CollectionInvalid:
            st.error('Error: The Collection referenced is invalid. Try again.')
        except pymongo.errors.PyMongoError:
            st.error('Error: Unknown fatal error. Execution failure point: Retrieving the database and collections'
                     'found in the database.')
        else:
            st.success('Successfully connected to the server!')

    def pull(self, query: dict):
        if query:
            self.query = {iter_: doc for (iter_, doc) in enumerate(self.collection.find(dict))}
            st.dataframe(pd.DataFrame.from_dict(self.query).astype(str))
        else:
            st.error('Error: Query cannot be empty.')

    def push(self, data: pandas.DataFrame):
        """
        Pushes data from local machine to the MongoDB server

        :param data:                A pandas DataFrame containing data of interest
        """

        if isinstance(data, pandas.DataFrame):
            data.reset_index(inplace=True)

            # dataframe is converted to records form as mongodb is a non-relational database
            data_ = data.to_dict('records')

            try:
                self.pointerCollection.insert_many(data_)
            except pymongo.errors.BulkWriteError:
                st.error('Error: Error in writing entire dataframe. Try again.')
            except pymongo.errors.DocumentTooLarge:
                st.error('Error: Document is too large. Try reducing the size of your document before uploading it.')
            except pymongo.errors.DuplicateKeyError:
                st.error('Error: Duplicate keys exist within dataset. Remove them before uploading to MongoDB.')
            except pymongo.errors.OperationFailure:
                st.error('Operation failed to execute. Try again.')
            else:
                st.success('Insertion to DB succeeded!')
        else:
            st.error(f'Error: Invalid datatype. Expected <pandas.DataFrame> but received {type(data)} instead.')

    def resetPointer(self, db: str, collections: str):
        """
        Resets the pointers to the database and collection of interest

        :param db:                      Database to change to
        :param collections:             Collection to change to
        """

        self.db = db
        self.collection = collections
        self.point()

    def close(self):
        self.authenticatedUser.stop()
        st.success('Database Connection Terminated!')
