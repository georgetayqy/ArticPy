"""
Load, Clean and Visualise is one of the core modules of this app.

This module is responsible for the loading, cleaning and visualising of the data to be used for further NLP analysis in
the other modules in this app.

The loading of data part is handled by pandas, while the cleaning part is largely being handled by Texthero. The
visualisation part is handled by streamlit, streamlit_pandas_profiling/pandas_profiling and pandas.
"""

# -------------------------------------------------------------------------------------------------------------------- #
# |                                         IMPORT RELEVANT LIBRARIES                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
import os
import pathlib
import nltk
import numpy as np
import openpyxl
import pandas as pd
import spacy
import streamlit as st
import texthero as hero
import pandas_profiling

from streamlit_pandas_profiling import st_profile_report
from texthero import preprocessing
from utils import csp_downloaders
from utils.helper import readFile, lemmatizeText, downloadCorpora

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                  INITIAL SETUP                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
DOWNLOAD_PATH = (STREAMLIT_STATIC_PATH / "downloads")

# DOWNLOAD THE NECESSARY CORPUS
downloadCorpora('words')
downloadCorpora('wordnet')
downloadCorpora('vader_lexicon')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                            GLOBAL VARIABLE DEFINITION                                            | #
# -------------------------------------------------------------------------------------------------------------------- #
FILE = 'Small File(s)'
MODE = 'CSV'
DATA_PATH = None
DATA = pd.DataFrame()
CSP = None
CLEAN = False
CLEAN_MODE = 'Simple'
SAVE = False
VERBOSE = False
VERBOSITY = False
CLEANED_DATA = pd.DataFrame()
CLEANED_DATA_TOKENIZED = pd.DataFrame()
ADVANCED_ANALYSIS = False
SIMPLE_PIPELINE = [
    preprocessing.remove_html_tags,
    preprocessing.remove_diacritics,
    preprocessing.remove_whitespace,
    preprocessing.remove_urls,
    preprocessing.drop_no_content
]
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
# APPLY preprocessing.remove_digits(only_blocks=False) TO THE CUSTOM PIPELINE AFTER CLEANING
FINALISED_DATA_LIST = []
DATA_COLUMN = None
QUERY = None


# -------------------------------------------------------------------------------------------------------------------- #
# |                                              MAIN APP FUNCTIONALITY                                              | #
# -------------------------------------------------------------------------------------------------------------------- #
def app():
    """
    Main function that will be called when the app is run and this module is called
    """

# -------------------------------------------------------------------------------------------------------------------- #
# |                                               GLOBAL VARIABLES                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    global FILE, MODE, DATA_PATH, DATA, CSP, CLEAN, CLEAN_MODE, SAVE, VERBOSE, VERBOSITY, CLEANED_DATA, \
        CLEANED_DATA_TOKENIZED, SIMPLE_PIPELINE, ADVANCED_ANALYSIS, FINALISED_DATA_LIST, DATA_COLUMN, QUERY

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                    INIT                                                          | #
# -------------------------------------------------------------------------------------------------------------------- #
    # CREATE THE DOWNLOAD PATH WHERE FILES CREATED WILL BE STORED AND AVAILABLE FOR DOWNLOADING
    if not DOWNLOAD_PATH.is_dir():
        DOWNLOAD_PATH.mkdir()

    st.title('Load, Clean and Visualise Data')
    st.markdown('## Init\n'
                'This module is used for the visualisation and cleaning of data used for NLP Analysis on news '
                'articles. Since this process uses the nltk stopword corpus, you need to download the corpus onto your '
                'system. The corpus will automatically download onto your device when you run the app. Please ensure '
                'that sufficient storage (~1 GB) of storage space is free, and that you are connected to the Internet '
                'to ensure that the corpus can be successfully downloaded. If the download fails, rerun the app again '
                'and ensure that your device has sufficient space and is connected to the Internet.\n\n '
                'For the cleaning process, all non-ASCII characters will be removed, and all non-English text '
                'will be removed. Multi-language support has not been implemented into this module as of yet.\n\n'
                '## Upload Data\n'
                'Due to limitations imposed by the file uploader widget, only files smaller than 200 MB can be loaded '
                'with the widget. If your file is larger than 200 MB, you may choose to rerun the app with the tag '
                '`--server.maxUploadSize=[SIZE_IN_MB_HERE]` appended behind the `streamlit run app.py` command and '
                'define the maximum size of file you can upload onto Streamlit, or use the Large File option to pull '
                'your dataset from any one of the three supported Cloud Service Providers into the app. Note that '
                'modifying the command you use to run the app is not available if you are using the web interface for '
                'the app and you will be limited to using the Large File option to pull datasets larger than 200 MB '
                'in size. For Docker, you will need to append the tag above behind the Docker Image name when running '
                'the *run* command, e.g. `docker run asdfghjklxl/news:latest --server.maxUploadSize=1028`.\n\n'
                'Select the file format you wish to upload. You are warned that if you fail to define the correct file '
                'format you wish to upload, the app will not let you upload it (if you are using the Small File module '
                'and may result in errors (for Large File module).\n\n')
    FILE = st.selectbox('Select the type of file to load', ('Select File Mode', 'Small File(s)', 'Large File(s)'))
    MODE = st.selectbox('Define the data input format', ('CSV', 'XLSX'))

# -------------------------------------------------------------------------------------------------------------------- #
# |                                               PROCESSING FLAGS                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    st.markdown('## Flags')
    CLEAN = st.checkbox('Clean the data?')
    SAVE = st.checkbox('Output to CSV file?')
    VERBOSE = st.checkbox('Print out DataFrames?')
    if VERBOSE:
        VERBOSITY = st.slider('Data points',
                              key='Data points to display?',
                              min_value=1,
                              max_value=1000,
                              value=20)
        ADVANCED_ANALYSIS = st.checkbox('Display Advanced DataFrame Statistics?')
    if CLEAN:
        st.markdown('## Preprocessing\n'
                    'The below options will allow you to specify whether to conduct a simple cleaning ('
                    'sentence structures and context is retained, while any wrongly encoded characters will be '
                    'removed), a complex cleaning (a process to lemmatize words and remove stopwords) and '
                    'advanced cleaning (advanced NLP techniques will be used to process the data).')
        CLEAN_MODE = st.selectbox('Select Preprocessing Pipelines', ('Simple', 'Complex'))

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                 FILE UPLOADING                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    if FILE == 'Small File(s)':
        st.markdown('### Upload the file that you wish to analyse:\n')
        DATA_PATH = st.file_uploader(f'Load up a {MODE} File containing the cleaned data', type=[MODE])
        if DATA_PATH is not None:
            DATA = readFile(DATA_PATH, MODE)
            if not DATA.empty or not DATA:
                DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                st.success('Data Loaded!')

    elif FILE == 'Large File(s)':
        st.info(f'File Format Selected: {MODE}')
        CSP = st.selectbox('CSP', ('Select a CSP', 'Azure', 'Amazon', 'Google', 'Google Drive'))

        if CSP == 'Azure':
            azure = csp_downloaders.AzureDownloader()
            if azure.SUCCESSFUL:
                try:
                    azure.downloadBlob()
                    DATA = readFile(azure.AZURE_DOWNLOAD_ABS_PATH, MODE)
                    if not DATA.empty:
                        DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                        st.info('File Read!')
                except Exception as ex:
                    st.error(f'Error: {ex}. Try again.')

        elif CSP == 'Amazon':
            aws = csp_downloaders.AWSDownloader()
            if aws.SUCCESSFUL:
                try:
                    aws.downloadFile()
                    DATA = readFile(aws.AWS_FILE_NAME, MODE)
                    if not DATA.empty:
                        DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                        st.info('File Read!')
                except Exception as ex:
                    st.error(f'Error: {ex}. Try again.')

        elif CSP == 'Google':
            gcs = csp_downloaders.GoogleDownloader()
            if gcs.SUCCESSFUL:
                try:
                    gcs.downloadBlob()
                    DATA = readFile(gcs.GOOGLE_DESTINATION_FILE_NAME, MODE)
                    if not DATA.empty:
                        DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                        st.info('File Read!')
                except Exception as ex:
                    st.error(f'Error: {ex}. Try again.')

        elif CSP == 'Google Drive':
            gd = csp_downloaders.GoogleDriveDownloader()
            if gd.SUCCESSFUL:
                try:
                    gd.downloadBlob()
                    DATA = readFile(gd.GOOGLE_DRIVE_OUTPUT_FILENAME, MODE)
                    if not DATA.empty:
                        DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                        st.info('File Read!')
                except Exception as ex:
                    st.error(f'Error: {ex}. Try again.')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                           DATA LOADING AND PROCESSING                                            | #
# -------------------------------------------------------------------------------------------------------------------- #
    st.markdown('## Analysis Operation\n'
                'Ensure that you have successfully uploaded the required data before clicking on the "Begin Analysis" '
                'button.')

    if st.button('Begin Analysis', key='runner'):
        if DATA_PATH:
            try:
                DATA = DATA[[DATA_COLUMN]]
                DATA[DATA_COLUMN] = DATA[DATA_COLUMN].str.encode('ascii', 'ignore').str.decode('ascii')
                DATA = pd.DataFrame(data=DATA)
                DATA = DATA.dropna()
            except Exception as ex:
                st.error(f'Error: {ex}')
            else:
                st.info('Data parsed!')

# -------------------------------------------------------------------------------------------------------------------- #
# +                                                 DATA CLEANER                                                     + #
# -------------------------------------------------------------------------------------------------------------------- #
            if not DATA.empty:
                if CLEAN:
                    if CLEAN_MODE == 'Simple':
                        try:
                            CLEANED_DATA = DATA[[DATA_COLUMN]]

                            # PREPROCESSING AND CLEANING
                            CLEANED_DATA['CLEANED CONTENT'] = hero.clean(CLEANED_DATA[DATA_COLUMN], SIMPLE_PIPELINE)
                            CLEANED_DATA['CLEANED CONTENT'].replace('', np.nan, inplace=True)
                            CLEANED_DATA.dropna(inplace=True, subset=['CLEANED CONTENT'])
                            CLEANED_DATA = CLEANED_DATA.astype(str).to_frame()
                        except Exception as ex:
                            st.error(ex)

                    elif CLEAN_MODE == 'Complex' or CLEAN_MODE == 'Advanced':
                        try:
                            CLEANED_DATA = DATA[[DATA_COLUMN]]

                            # PREPROCESSING AND CLEANING
                            CLEANED_DATA['CLEANED CONTENT'] = hero.clean(CLEANED_DATA[DATA_COLUMN], PIPELINE)
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
                            CLEANED_DATA_TOKENIZED = CLEANED_DATA_TOKENIZED.to_frame().astype(str)

                        except Exception as ex:
                            st.error(ex)

                FINALISED_DATA_LIST = [
                    (DATA, 'Raw Data', 'raw_ascii_data.csv'),
                    (CLEANED_DATA, 'Cleaned Data', 'cleaned_data.csv'),
                    (CLEANED_DATA_TOKENIZED, 'Cleaned Tokenized Data', 'tokenized.csv')
                ]

# -------------------------------------------------------------------------------------------------------------------- #
# +                                        VISUALISE THE DATA: CLEANED DATA                                          + #
# -------------------------------------------------------------------------------------------------------------------- #
                if VERBOSE:
                    if CLEAN:
                        if 'CLEANED CONTENT' in CLEANED_DATA.columns:
                            if VERBOSITY != 0:
                                try:
                                    for data in FINALISED_DATA_LIST:
                                        if not data[0].empty:
                                            st.markdown(f'## {data[1]}')
                                            st.dataframe(data[0].head(VERBOSITY), height=400, width=800)

                                            if ADVANCED_ANALYSIS:
                                                with st.expander('Advanced Profile Report'):
                                                    st_profile_report(data[0].profile_report(
                                                        explorative=True,
                                                        minimal=True
                                                    ))
                                except RuntimeError:
                                    st.warning(
                                        'Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                                    for data in FINALISED_DATA_LIST:
                                        if not data[0].empty:
                                            st.markdown(f'## {data[1]}')
                                            st.dataframe(data[0].head(10), height=400, width=800)

                                            if ADVANCED_ANALYSIS:
                                                with st.expander('Advanced Profile Report'):
                                                    st_profile_report(data[0].profile_report(
                                                        explorative=True,
                                                        minimal=True))
                                except KeyError:
                                    st.error('Warning: Your data was not processed properly. Try again.')

                            else:
                                try:
                                    for data in FINALISED_DATA_LIST:
                                        if not data[0].empty:
                                            st.markdown(f'## {data[1]}')
                                            st.dataframe(data[0], height=400, width=800)

                                            if ADVANCED_ANALYSIS:
                                                with st.expander('Advanced Profile Report'):
                                                    st_profile_report(data[0].profile_report(
                                                        explorative=True,
                                                        minimal=True))
                                except RuntimeError:
                                    st.warning(
                                        'Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                                    for data in FINALISED_DATA_LIST:
                                        if not data[0].empty:
                                            st.markdown(f'## {data[1]}')
                                            st.dataframe(data[0].head(10), height=400, width=800)

                                            if ADVANCED_ANALYSIS:
                                                with st.expander('Advanced Profile Report'):
                                                    st_profile_report(data[0].profile_report(
                                                        explorative=True,
                                                        minimal=True))
                                except KeyError:
                                    st.error('Warning: Your data was not processed properly. Try again.')
                        else:
                            st.error('Error: KeyError -> CLEANED CONTENT column is missing data. Check your data '
                                     'source to ensure that CONTENT is the column header and that there is data '
                                     'in the column.')

# -------------------------------------------------------------------------------------------------------------------- #
# +                                          VISUALISE THE DATA: RAW DATA                                            + #
# -------------------------------------------------------------------------------------------------------------------- #
                    else:
                        if VERBOSITY != 0:
                            try:
                                if not DATA.empty:
                                    st.markdown('## DataFrame Output')
                                    st.dataframe(DATA.head(VERBOSITY), height=400, width=800)

                                    if ADVANCED_ANALYSIS:
                                        with st.expander('Advanced Profile Report'):
                                            st_profile_report(DATA[[DATA_COLUMN]].profile_report(
                                                explorative=True,
                                                minimal=True))
                            except RuntimeError:
                                st.warning(
                                    'Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                                st.dataframe(DATA.head(10), height=400, width=800)

                                if ADVANCED_ANALYSIS:
                                    with st.expander('Advanced Profile Report'):
                                        st_profile_report(DATA[[DATA_COLUMN]].profile_report(
                                            explorative=True,
                                            minimal=True))
                            except KeyError:
                                st.error('Warning: Your data was not processed properly. Try again.')

                        else:
                            try:
                                st.markdown('## DataFrame Output')
                                st.dataframe(DATA, height=400, width=800)

                                if ADVANCED_ANALYSIS:
                                    with st.expander('Advanced Profile Report'):
                                        st_profile_report(DATA[[DATA_COLUMN]].profile_report(
                                            explorative=True,
                                            minimal=True))
                            except RuntimeError:
                                st.warning(
                                    'Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                                st.dataframe(DATA.head(10), height=400, width=800)

                                if ADVANCED_ANALYSIS:
                                    with st.expander('Advanced Profile Report'):
                                        st_profile_report(DATA[[DATA_COLUMN]].profile_report(
                                            explorative=True,
                                            minimal=True))
                            except KeyError:
                                st.error('Warning: Your data was not processed properly. Try again.')

# -------------------------------------------------------------------------------------------------------------------- #
# +                                                   SAVE THE DATA                                                  + #
# -------------------------------------------------------------------------------------------------------------------- #
                if SAVE:
                    st.markdown('## Download Data')
                    try:
                        for data in FINALISED_DATA_LIST:
                            if not data[0].empty:
                                st.markdown(f'### {data[1]}\n'
                                            f'Download data from [downloads/{data[2]}]'
                                            f'(downloads/{data[2]})')
                                data[0].to_csv(str(DOWNLOAD_PATH / f'{data[2]}'), index=False)
                    except KeyError:
                        st.error('Warning: Your data was not processed properly. Try again.')
                    except Exception as ex:
                        st.error(f'Error: Unknown Fatal Error -> {ex}')
            else:
                st.error('Error: No files uploaded.')
        else:
            st.error('Error: No files uploaded.')
