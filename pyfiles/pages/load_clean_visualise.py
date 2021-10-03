"""
This module is used for loading, cleaning (optional) and visualising of the raw data or processed data passed on to it
by the user. This module uses the TextHero and NLTK package to conduct the elementary NLP pre-processing processes such
as removal of any non-alphabetical symbols and lemmatizing the words.
"""

# -------------------------------------------------------------------------------------------------------------------- #
# |                                         IMPORT RELEVANT LIBRARIES                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
import pathlib

import nltk
import numpy as np
import openpyxl
import pandas as pd
import spacy

import streamlit as st
import texthero as hero
from texthero import preprocessing

from utils import csp_downloaders
from utils.helper import readFile, lemmatizeText

# -------------------------------------------------------------------------------------------------------------------- #
# |                                               BASE DOWNLOAD PATH                                                 | #
# -------------------------------------------------------------------------------------------------------------------- #
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
DOWNLOAD_PATH = (STREAMLIT_STATIC_PATH / "downloads")

# -------------------------------------------------------------------------------------------------------------------- #
# |                                CUSTOM PREPROCESSING PIPELINES AND REDUNDANT FLAGS                                | #
# -------------------------------------------------------------------------------------------------------------------- #
# APPLY preprocessing.remove_digits(only_blocks=False) TO THE CUSTOM PIPELINE AFTER
PIPELINE = [
    preprocessing.fillna,
    preprocessing.lowercase,
    preprocessing.remove_punctuation,
    preprocessing.remove_html_tags,
    preprocessing.remove_diacritics,
    preprocessing.remove_stopwords,
    preprocessing.remove_whitespace,
    preprocessing.remove_urls,
    preprocessing.drop_no_content
]

SIMPLE_PIPELINE = [
    preprocessing.remove_html_tags,
    preprocessing.remove_diacritics,
    preprocessing.remove_whitespace,
    preprocessing.remove_urls,
    preprocessing.drop_no_content
]

DATA = pd.DataFrame()
CLEANED_DATA = pd.DataFrame()
CLEANED_DATA_TOKENIZED = pd.DataFrame()
TF_IDF_FILE = pd.DataFrame()
K_MEANS_FILE = pd.DataFrame()
PCA_FILE = pd.DataFrame()
DATA_PATH = None

# DOWNLOAD THE NECESSARY CORPUS
nltk.download('words')


# -------------------------------------------------------------------------------------------------------------------- #
# |                                              MAIN APP FUNCTIONALITY                                              | #
# -------------------------------------------------------------------------------------------------------------------- #
def app():
    """
    Main function that will be called when the app is run
    """

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                             GLOBAL VARIABLES                                                 | #
    # ---------------------------------------------------------------------------------------------------------------- #
    global FILE, MODE, DATA_PATH, DATA, CSP, CLEAN, CLEAN_MODE, SAVE, VERBOSE, VERBOSITY, CLEANED_DATA, TOP_WORDS, \
        TOP_WORDS_VERBOSE, CLEANED_DATA_TOKENIZED, TF_IDF, K_MEANS, N_CLUSTER, PCA, SIMPLE_PIPELINE, TF_IDF_DATA, \
        K_MEANS_DATA, PCA_DATA

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                                  INIT                                                        | #
    # ---------------------------------------------------------------------------------------------------------------- #
    # CREATE THE DOWNLOAD PATH WHERE FILES CREATED WILL BE STORED AND AVAILABLE FOR DOWNLOADING
    if not DOWNLOAD_PATH.is_dir():
        DOWNLOAD_PATH.mkdir()

    st.title('Load, Clean and Visualise Data')
    st.markdown('## Init\n'
                'Since this process uses the nltk stopword corpus, you need to download the corpus onto your system. '
                'To do so, activate the your previously created Conda environment on your terminal or IDE. Then, key '
                'in the following commands:')
    st.code('>>> python\n'
            '>>> import texthero\n'
            '>>> exit()\n')
    st.markdown('The above code will then pull the corpus and download it onto your machine and exit once you are '
                'done. For the cleaning process, all non-ASCII characters will be removed, and all non-English text '
                'will be removed. Multi-language support has not been implemented into this module as of yet.')
    st.markdown('## Upload Data\n'
                'Due to limitations imposed by the file uploader widget, only files smaller than 200 MB can be loaded '
                'with the widget. If your file is larger than 200 MB, you may choose to rerun the app with the tag '
                '`--server.maxUploadSize=[SIZE_IN_MB_HERE]` appended behind the `streamlit run app.py` command and '
                'define the maximum size of file you can upload onto Streamlit, or use the Large File option to pull '
                'your dataset from any one of the three supported Cloud Service Providers into the app. Note that '
                'modifying the command you use to run the app is not available if you are using the web interface for '
                'the app and you will be limited to using the Large File option to pull datasets larger than 200 MB '
                'in size.\n\n'
                'Next, select the file format you wish to upload.\n\n'
                'Use the "CONTENT" template on the left to key in and process your data. If you do not wish to use '
                'the template for any reason, ensure that the column name "CONTENT" is in your dataset, and that '
                'your data is stored in the column under that name.')
    FILE = st.selectbox('Select the type of file to load', ('Select File Mode', 'Small File(s)', 'Large File(s)'))
    MODE = st.selectbox('Define the data input format', ('CSV', 'XLSX'))

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                             PROCESSING FLAGS                                                 | #
    # ---------------------------------------------------------------------------------------------------------------- #
    st.markdown('## Flags')
    CLEAN = st.checkbox('Clean the data?')
    SAVE = st.checkbox('Output to CSV file?')
    VERBOSE = st.checkbox('Print out DataFrames?')
    if VERBOSE:
        VERBOSITY = st.slider('Data points', key='Data points to display?', min_value=1, max_value=1000, value=20)
    if CLEAN:
        st.markdown('## Preprocessing\n'
                    'The below options will allow you to specify whether to conduct a simple cleaning (whereby '
                    'sentence structures and context is retained, while any wrongly encoded characters will be '
                    'removed.), a complex cleaning (a process to lemmatize words and remove stopwords) and '
                    'advanced cleaning (advanced NLP techniques will be employed to process the data).')
        CLEAN_MODE = st.selectbox('Select Preprocessing Pipelines', ('Simple', 'Complex', 'Advanced'))
        if CLEAN_MODE == 'Advanced':
            st.markdown('### Advanced Processing\n'
                        'The following section details advanced text analysis methods such as Term Frequency-Inverse '
                        'Document Frequency and K-means Analysis. You are warned that your system must be configured '
                        'in a way to allow TextHero to use a significant amount of disk space as a paging file or it '
                        'may run into MemoryError exceptions, which occurs due to the system not being able to '
                        'allocate sufficient memory to process the data. However, this is largely dependent on the '
                        'size of the dataset you want to process. From our experience, a dataset containing 100k '
                        'news articles required a paging file size of around 400 GiB, though it may differ from your '
                        'use case.')
            TF_IDF = st.checkbox('Term Frequency-Inverse Document Frequency')
            K_MEANS = st.checkbox('K-means Analysis')
            if K_MEANS:
                N_CLUSTER = st.number_input('Maximum Neighbours for K-means Analysis',
                                            min_value=1,
                                            max_value=100,
                                            value=2)
                if TF_IDF:
                    PCA = st.checkbox('PCA Analysis')

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                               FILE UPLOADING                                                 | #
    # ---------------------------------------------------------------------------------------------------------------- #
    if FILE == 'Small File(s)':
        st.markdown('### Upload the file that you wish to analyse:\n')
        DATA_PATH = st.file_uploader('Load up a CSV/XLSX File containing the cleaned data', type=[MODE])
        if st.button('Load Data', key='Loader'):
            DATA = readFile(DATA_PATH, MODE)
            if not DATA.empty:
                st.success('Data Loaded!')

    elif FILE == 'Large File(s)':
        CSP = st.selectbox('CSP', ('None', 'Azure', 'Amazon', 'Google'))

        if CSP == 'Azure':
            azure = csp_downloaders.AzureDownloader()
            if st.button('Load Data', key='az'):
                azure.downloadBlob()
                DATA = readFile(csp_downloaders.AZURE_DOWNLOAD_ABS_PATH, MODE)
                st.info('File Read!')

        elif CSP == 'Amazon':
            aws = csp_downloaders.AWSDownloader()
            if st.button('Load Data', key='aws'):
                aws.downloadFile()
                DATA = readFile(csp_downloaders.AWS_FILE_NAME, MODE)
                st.info('File Read!')

        elif CSP == 'Google':
            gcs = csp_downloaders.GoogleDownloader()
            if st.button('Load Data', key='gcs'):
                gcs.downloadBlob()
                DATA = readFile(csp_downloaders.GOOGLE_DESTINATION_FILE_NAME, MODE)
                st.info('File Read!')

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                         DATA LOADING AND PROCESSING                                          | #
    # ---------------------------------------------------------------------------------------------------------------- #
    if st.button('Analyse Data', key='runner'):
        if DATA_PATH:
            try:
                DATA = DATA[['CONTENT']]
                DATA = pd.DataFrame(data=DATA['CONTENT'].str.encode('ascii', 'ignore').str.decode('ascii'))
                DATA = DATA.dropna()
            except Exception as e:
                st.error(f'Error: {e}')
            else:
                st.info('Data parsed!')

            if not DATA.empty:
                # ---------------------------------------------------------------------------------------------------- #
                # +                                           DATA CLEANER                                           + #
                # ---------------------------------------------------------------------------------------------------- #
                if CLEAN:
                    if CLEAN_MODE == 'Simple':
                        try:
                            CLEANED_DATA = DATA[["CONTENT"]]

                            # PREPROCESSING AND CLEANING
                            CLEANED_DATA['CLEANED CONTENT'] = hero.clean(CLEANED_DATA['CONTENT'], SIMPLE_PIPELINE)
                            CLEANED_DATA['CLEANED CONTENT'].replace('', np.nan, inplace=True)
                            CLEANED_DATA.dropna(inplace=True, subset=['CLEANED CONTENT'])
                            CLEANED_DATA = CLEANED_DATA.astype(str)
                        except Exception as e:
                            st.error(e)

                    elif CLEAN_MODE == 'Complex' or CLEAN_MODE == 'Advanced':
                        try:
                            CLEANED_DATA = DATA[["CONTENT"]]

                            # PREPROCESSING AND CLEANING
                            CLEANED_DATA['CLEANED CONTENT'] = hero.clean(CLEANED_DATA['CONTENT'], PIPELINE)
                            CLEANED_DATA['CLEANED CONTENT'] = hero.remove_digits(CLEANED_DATA['CLEANED CONTENT'],
                                                                                 only_blocks=False)
                            CLEANED_DATA['CLEANED CONTENT'].replace('', np.nan, inplace=True)
                            CLEANED_DATA.dropna(subset=['CLEANED CONTENT'], inplace=True)

                            # PRELIMINARY TOKENIZATION THE DATA
                            CLEANED_DATA_TOKENIZED = hero.tokenize(CLEANED_DATA['CLEANED CONTENT'])
                            CLEANED_DATA_TOKENIZED = CLEANED_DATA_TOKENIZED.apply(lemmatizeText)

                            # SPELL CHECK
                            words = set(word.lower() for word in nltk.corpus.words.words())
                            for index, row in CLEANED_DATA_TOKENIZED.iteritems():
                                CLEANED_DATA.at[index, 'CLEANED CONTENT'] = \
                                    (" ".join(word for word in row if word.lower() in words or not word.isalpha()))
                            CLEANED_DATA['CLEANED CONTENT'].replace('', np.nan, inplace=True)
                            CLEANED_DATA.dropna(subset=['CLEANED CONTENT'], inplace=True)
                            CLEANED_DATA = CLEANED_DATA.astype(str)

                            # FINAL TOKENIZATION THE DATA
                            CLEANED_DATA_TOKENIZED = hero.tokenize(CLEANED_DATA['CLEANED CONTENT'])
                            CLEANED_DATA_TOKENIZED = CLEANED_DATA_TOKENIZED.astype(str)

                            # ADVANCED PREPROCESSING
                            if CLEAN_MODE == 'Advanced':
                                if TF_IDF:
                                    CLEANED_DATA['TF-IDF'] = (
                                        CLEANED_DATA['CLEANED CONTENT']
                                        .pipe(hero.tfidf)
                                    )
                                    TF_IDF_DATA = CLEANED_DATA[['TF-IDF']]
                                if K_MEANS:
                                    CLEANED_DATA['K-MEANS'] = (
                                        CLEANED_DATA['CLEANED CONTENT']
                                        .pipe(hero.representation.kmeans, n_cluster=N_CLUSTER)
                                    )
                                    K_MEANS_DATA = CLEANED_DATA[['K-MEANS']]
                                if TF_IDF and K_MEANS:
                                    if PCA:
                                        CLEANED_DATA['PCA'] = (
                                            CLEANED_DATA['TF-IDF']
                                            .pipe(hero.pca)
                                        )
                                    PCA_DATA = CLEANED_DATA[['PCA']]
                        except Exception as e:
                            st.error(e)

                # ---------------------------------------------------------------------------------------------------- #
                # +                                        VISUALISE THE DATA                                        + #
                # ---------------------------------------------------------------------------------------------------- #
                if VERBOSE:
                    if CLEAN:
                        if 'CLEANED CONTENT' in CLEANED_DATA.columns:
                            if VERBOSITY != 0:
                                try:
                                    if not CLEANED_DATA.empty:
                                        st.markdown('## Cleaned Data DataFrame')
                                        st.dataframe(CLEANED_DATA['CLEANED CONTENT'].head(VERBOSITY))
                                    if not CLEANED_DATA_TOKENIZED.empty:
                                        st.markdown('## Cleaned Tokenized DataFrame')
                                        st.dataframe(CLEANED_DATA_TOKENIZED.head(VERBOSITY))
                                    if not TF_IDF_DATA.empty:
                                        st.markdown('## TF-IDF DataFrame')
                                        st.dataframe(TF_IDF_DATA.head(VERBOSITY))
                                    if not K_MEANS_DATA.empty:
                                        st.markdown('## K-Means DataFrame')
                                        st.dataframe(K_MEANS_DATA.head(VERBOSITY))
                                    if not PCA_DATA.empty:
                                        st.markdown('## PCA DataFrame')
                                        st.dataframe(K_MEANS_DATA.head(VERBOSITY))
                                except RuntimeError:
                                    st.warning(
                                        'Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                                    if not CLEANED_DATA.empty:
                                        st.markdown('## Cleaned Data DataFrame')
                                        st.dataframe(CLEANED_DATA['CLEANED CONTENT'].head(10))
                                    if not CLEANED_DATA_TOKENIZED.empty:
                                        st.markdown('## Cleaned Tokenized DataFrame')
                                        st.dataframe(CLEANED_DATA_TOKENIZED.head(10))
                                    if not TF_IDF_DATA.empty:
                                        st.markdown('## TF-IDF DataFrame')
                                        st.dataframe(TF_IDF_DATA.head(10))
                                    if not K_MEANS_DATA.empty:
                                        st.markdown('## K-Means DataFrame')
                                        st.dataframe(K_MEANS_DATA.head(10))
                                    if not PCA_DATA.empty:
                                        st.markdown('## PCA DataFrame')
                                        st.dataframe(K_MEANS_DATA.head(10))
                                except KeyError:
                                    st.error('Warning: Your data was not processed properly. Try again.')

                            else:
                                try:
                                    if not CLEANED_DATA.empty:
                                        st.markdown('## Cleaned Data DataFrame')
                                        st.dataframe(CLEANED_DATA['CLEANED CONTENT'])
                                    if not CLEANED_DATA_TOKENIZED.empty:
                                        st.markdown('## Cleaned Tokenized DataFrame')
                                        st.dataframe(CLEANED_DATA_TOKENIZED)
                                    if not TF_IDF_DATA.empty:
                                        st.markdown('## TF-IDF DataFrame')
                                        st.dataframe(TF_IDF_DATA)
                                    if not K_MEANS_DATA.empty:
                                        st.markdown('## K-Means DataFrame')
                                        st.dataframe(K_MEANS_DATA)
                                    if not PCA_DATA.empty:
                                        st.markdown('## PCA DataFrame')
                                        st.dataframe(K_MEANS_DATA)
                                except RuntimeError:
                                    st.warning(
                                        'Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                                    if not CLEANED_DATA.empty:
                                        st.markdown('## Cleaned Data DataFrame')
                                        st.dataframe(CLEANED_DATA['CLEANED CONTENT'].head(10))
                                    if not CLEANED_DATA_TOKENIZED.empty:
                                        st.markdown('## Cleaned Tokenized DataFrame')
                                        st.dataframe(CLEANED_DATA_TOKENIZED.head(10))
                                    if not TF_IDF_DATA.empty:
                                        st.markdown('## TF-IDF DataFrame')
                                        st.dataframe(TF_IDF_DATA.head(10))
                                    if not K_MEANS_DATA.empty:
                                        st.markdown('## K-Means DataFrame')
                                        st.dataframe(K_MEANS_DATA.head(10))
                                    if not PCA_DATA.empty:
                                        st.markdown('## PCA DataFrame')
                                        st.dataframe(K_MEANS_DATA.head(10))
                                except KeyError:
                                    st.error('Warning: Your data was not processed properly. Try again.')
                        else:
                            st.error('Error: KeyError -> CLEANED CONTENT column is missing data. Check your data '
                                     'source to ensure that CONTENT is the column header and that there is data '
                                     'in the column.')
                    else:
                        if 'CONTENT' in DATA.columns:
                            if VERBOSITY != 0:
                                try:
                                    if not DATA.empty:
                                        st.markdown('## DataFrame Output')
                                        st.dataframe(DATA.head(VERBOSITY))
                                except RuntimeError:
                                    st.warning(
                                        'Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                                    st.dataframe(DATA.head(10))
                                except KeyError:
                                    st.error('Warning: Your data was not processed properly. Try again.')
                            else:
                                try:
                                    st.markdown('## DataFrame Output')
                                    st.dataframe(DATA)
                                except RuntimeError:
                                    st.warning(
                                        'Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                                    st.dataframe(DATA.head(10))
                                except KeyError:
                                    st.error('Warning: Your data was not processed properly. Try again.')
                        else:
                            st.error('Error: KeyError -> CONTENT column is missing data. Check your data source to '
                                     'ensure that CONTENT is the column header and that there is data in the column.')
                    if CLEAN_MODE == 'Advanced':
                        if TF_IDF and K_MEANS and PCA:
                            st.markdown('## TF-IDF, K-means and PCA Visualisation')
                            st.write(hero.scatterplot(CLEANED_DATA, 'PCA', color='K-MEANS', title='K-means'))

                # ---------------------------------------------------------------------------------------------------- #
                # +                                           SAVE THE DATA                                          + #
                # ---------------------------------------------------------------------------------------------------- #
                if SAVE:
                    if CLEAN:
                        st.markdown('## Download Data')
                        try:
                            if not DATA.empty:
                                st.markdown('### Raw Data\n'
                                            'Download raw(ish) data from [downloads/raw_ascii_data.csv]'
                                            '(downloads/raw_ascii_data.csv)')
                                DATA.to_csv(str(DOWNLOAD_PATH / 'raw_ascii_data.csv'))
                            if not CLEANED_DATA.empty:
                                st.markdown('### Cleaned Data\n')
                                st.markdown('Download requested data from [downloads/cleaned_data.csv]'
                                            '(downloads/cleaned_data.csv)')
                                CLEANED_DATA['CLEANED CONTENT'].to_csv(str(DOWNLOAD_PATH / 'cleaned_data.csv'))
                            if not CLEANED_DATA_TOKENIZED.empty:
                                st.markdown('### Cleaned Tokenized Data\n'
                                            'Download tokenized data from [downloads/tokenized.csv]'
                                            '(downloads/tokenized.csv)')
                                CLEANED_DATA_TOKENIZED.to_csv(str(DOWNLOAD_PATH / 'tokenized.csv'))
                            if not TF_IDF_DATA.empty:
                                st.markdown('### TF-IDF Data\n'
                                            'Download TF-IDF data from [downloads/tf_idf.csv]'
                                            '(downloads/tf_idf.csv)')
                                TF_IDF_DATA.to_csv(str(DOWNLOAD_PATH / 'tf_idf.csv'))
                            if not K_MEANS_DATA.empty:
                                st.markdown('### K-means Data\n'
                                            'Download TF-IDF data from [downloads/k_means.csv]'
                                            '(downloads/k_means.csv)')
                                K_MEANS_DATA.to_csv(str(DOWNLOAD_PATH / 'k_means.csv'))
                            if not PCA_DATA.empty:
                                st.markdown('### PCA Data\n'
                                            'Download PCA data from [downloads/pca.csv]'
                                            '(downloads/pca.csv)')
                                PCA_DATA.to_csv(str(DOWNLOAD_PATH / 'pca.csv'))
                        except KeyError:
                            st.error('Warning: Your data was not processed properly. Try again.')
                        except Exception as e:
                            st.error(f'Error: Unknown Fatal Error -> {e}')
                    else:
                        st.markdown('## Download Data')
                        try:
                            if not DATA.empty:
                                st.markdown('### Raw Data\n'
                                            'Download raw(ish) data from [downloads/raw_ascii_data.csv]'
                                            '(downloads/raw_ascii_data.csv)')
                                DATA.to_csv(str(DOWNLOAD_PATH / 'raw_ascii_data.csv'))
                            else:
                                st.error('Warning: Your data was not processed properly. Try again.')
                        except KeyError:
                            st.error('Warning: Your data was not processed properly. Try again.')
                        except Exception as e:
                            st.error(f'Error: Unknown Fatal Error -> {e}')
            else:
                st.error('Error: No files uploaded.')
        else:
            st.error('Error: No files uploaded.')
