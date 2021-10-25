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
import pathlib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
import texthero as hero

from texthero import preprocessing
from utils import csp_downloaders
from utils.helper import readFile, lemmatizeText, downloadCorpora, printDataFrame

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                  INITIAL SETUP                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
DOWNLOAD_PATH = (STREAMLIT_STATIC_PATH / "downloads")

# # DOWNLOAD THE NECESSARY CORPUS
downloadCorpora('words')

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
TOKENIZE = True
EXTEND_STOPWORD = False
STOPWORD_LIST = str()
FIN_STOPWORD_LIST = []
FINALISE = False


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
        CLEANED_DATA_TOKENIZED, SIMPLE_PIPELINE, ADVANCED_ANALYSIS, FINALISED_DATA_LIST, DATA_COLUMN, QUERY, \
        TOKENIZE, EXTEND_STOPWORD, STOPWORD_LIST, FIN_STOPWORD_LIST, FINALISE

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
                'with the widget. To circumvent this limitation, you may choose to '
                'rerun the app with the tag `--server.maxUploadSize=[SIZE_IN_MB_HERE]` appended behind the '
                '`streamlit run app.py` command and define the maximum size of file you can upload '
                'onto Streamlit (replace `SIZE_IN_MB_HERE` with an integer value above). Do note that this option '
                'is only available for users who run the app using the app\'s source code, or through Docker. '
                'For Docker, you will need to append the tag above behind the Docker Image name when running the *run* '
                'command, e.g. `docker run asdfghjklxl/news:latest --server.maxUploadSize=1028`; if you do not, the '
                'app will run with a default maximum upload size of 200 MB.\n\n'
                'Alternatively, you may use the Large File option to pull your dataset from any one of the four '
                'supported Cloud Service Providers into the app.\n\n'
                'After selecting the size of your file, select the file format you wish to upload. You are warned '
                'that if you fail to define the correct file format you wish to upload, the app will not let you '
                'upload your file (if you are using the Small File option, only the defined file format will be '
                'accepted by the File Uploader widget) and may result in errors (for Large File option).\n\n')
    FILE = st.selectbox('Select the Size of File to Load', ('Small File(s)', 'Large File(s)'))
    MODE = st.selectbox('Define the Data Input Format', ('CSV', 'XLSX'))

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                 FILE UPLOADING                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    if FILE == 'Small File(s)':
        st.markdown('### Upload the File that you wish to Analyse:\n')
        DATA_PATH = st.file_uploader(f'Load {MODE} File', type=[MODE])
        if DATA_PATH is not None:
            DATA = readFile(DATA_PATH, MODE)
            if not DATA.empty or not DATA:
                DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                st.success(f'Data Loaded from {DATA_COLUMN}!')
        else:
            # RESET
            DATA = pd.DataFrame()

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
                        st.success(f'Data Loaded from {DATA_COLUMN}!')
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
                        st.success(f'Data Loaded from {DATA_COLUMN}!')
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
                        st.success(f'Data Loaded from {DATA_COLUMN}!')
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
                        st.success(f'Data Loaded from {DATA_COLUMN}!')
                except Exception as ex:
                    st.error(f'Error: {ex}. Try again.')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                               PROCESSING FLAGS                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    st.markdown('## Flags\n'
                'Note that there is an size limit **(50 MB)** for the DataFrames that are printed to screen. If '
                'you get an error telling you that the DataFrame size is too large to proceed, kindly lower the number '
                'of data points you wish to visualise or download the file and visualise it through Excel or any other '
                'DataFrame visualising Python packages. There is no definitive way to increase the size of the '
                'DataFrame that can be printed out due to the inherent limitation on the size of the packets sent '
                'over to and from the Streamlit server.')
    CLEAN = st.checkbox('Clean the data?')
    SAVE = st.checkbox('Save Output DataFrame into CSV File?')
    VERBOSE = st.checkbox('Print out DataFrames?')
    if VERBOSE:
        VERBOSITY = st.slider('Data Points To Print',
                              key='Data points to display?',
                              min_value=1,
                              max_value=1000,
                              value=20)
        ADVANCED_ANALYSIS = st.checkbox('Display Advanced DataFrame Statistics?',
                                        help='This option will analyse your DataFrame and display advanced statistics '
                                             'on it. Note that this will require some time and processing power to '
                                             'complete. Deselect this option if this functionality is not required.')
    if CLEAN:
        st.markdown('## Preprocessing\n'
                    'The below options will allow you to specify whether to conduct a **simple cleaning** ('
                    '*sentence structures and context is retained*, while any wrongly encoded characters will be '
                    'removed) or a **complex cleaning** (a process to *lemmatize words and remove stopwords*).\n\n'
                    'Files created from **simple cleaning** may be used for Summarization in *NLP Toolkit* while '
                    'files created from **complex cleaning** may be used for Document-Term Matrix Creation in '
                    '*Document-Term Matrix*. Note that for most NLP Processes, the ** complex cleaning** process '
                    'is recommended.')
        CLEAN_MODE = st.selectbox('Select Preprocessing Pipelines', ('Simple', 'Complex'))
        TOKENIZE = st.checkbox('Tokenize Data?', value=True, help='This option is enabled by default. Deselect this '
                                                                  'option if you do not want to tokenize your data.')
        if CLEAN_MODE == 'Complex':
            EXTEND_STOPWORD = st.checkbox('Extend List of Stopwords?',
                                          help='Select this option to extend the list of stopwords that will be used '
                                               'to clean your data.')
            if EXTEND_STOPWORD:
                STOPWORD_LIST = str()
                STOPWORD_LIST = st.text_area('Extended List of Stopwords',
                                             value='Key in a list of stopwords that you wish to append to the existing '
                                                   'list of stopwords. Delimit your words by commas, e.g. this, is, a, '
                                                   'sample, list ...')
                if len(STOPWORD_LIST) != 0:
                    st.info('Alert: Words detected.')
                else:
                    st.info('Alert: No Words detected')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                           DATA LOADING AND PROCESSING                                            | #
# -------------------------------------------------------------------------------------------------------------------- #
    st.markdown('## Data Cleaning and Visualisation\n'
                'Ensure that you have successfully uploaded the required data and selected the correct column '
                'containing your data before clicking on the "Begin Analysis" button. The status of your file '
                'upload is displayed below for your reference.')
    if DATA_PATH:
        st.info('File loaded.')
    else:
        st.warning('File has not been loaded.')

    if st.button('Begin Analysis', key='runner'):
        # RESET STATE
        CLEANED_DATA = pd.DataFrame()
        CLEANED_DATA_TOKENIZED = pd.DataFrame()

        if DATA_PATH:
            try:
                DATA = DATA[[DATA_COLUMN]]
                DATA[DATA_COLUMN] = DATA[DATA_COLUMN].str.encode('ascii', 'ignore').str.decode('ascii')
                DATA = pd.DataFrame(data=DATA)
                DATA = DATA.dropna()
            except Exception as ex:
                st.error(f'Error: {ex}')

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
                            CLEANED_DATA = CLEANED_DATA.astype(str)

                            if TOKENIZE:
                                CLEANED_DATA_TOKENIZED = hero.tokenize(CLEANED_DATA['CLEANED CONTENT'])
                                CLEANED_DATA_TOKENIZED = CLEANED_DATA_TOKENIZED.to_frame().astype(str)
                        except Exception as ex:
                            st.error(ex)

                    elif CLEAN_MODE == 'Complex' or CLEAN_MODE == 'Advanced':
                        # CHECK IF NEW STOPWORDS WERE ADDED TO THE LIST
                        if EXTEND_STOPWORD:
                            try:
                                if len(STOPWORD_LIST) != 0:
                                    STOPWORD_LIST = [word.strip().lower() for word in STOPWORD_LIST.split(sep=',')]
                                    FIN_STOPWORD_LIST = [word.lower() for word in nltk.corpus.words.words()]
                                    FIN_STOPWORD_LIST.extend(STOPWORD_LIST)
                                    FIN_STOPWORD_LIST = set(FIN_STOPWORD_LIST)
                                    st.info(f'Stopwords accepted: {[word for word in STOPWORD_LIST]}!')
                                    FINALISE = True
                                else:
                                    FINALISE = False
                                    raise ValueError('Length of Stopword List is 0. Try again.')
                            except Exception as ex:
                                st.error(f'Error: {ex}')

                        # IF NO NEW STOPWORDS WERE DEFINED, THEN USE DEFAULT
                        else:
                            STOPWORD_LIST = str()
                            FIN_STOPWORD_LIST = set(word.lower() for word in nltk.corpus.words.words())
                            st.info('Alert: Default set of stopwords will be used.')
                            FINALISE = True

                        # NO ELSE CONDITION AS ELSE CONDITION IS EXPLICITLY SPECIFIED IN THE PREVIOUS EXCEPTION/ERROR
                        if FINALISE:
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
                                for index, row in CLEANED_DATA_TOKENIZED.iteritems():
                                    CLEANED_DATA.at[index, 'CLEANED CONTENT'] = \
                                        (" ".join(word for word in row if word.lower() in FIN_STOPWORD_LIST
                                                  or not word.isalpha()))
                                CLEANED_DATA['CLEANED CONTENT'].replace('', np.nan, inplace=True)
                                CLEANED_DATA.dropna(subset=['CLEANED CONTENT'], inplace=True)
                                CLEANED_DATA = CLEANED_DATA.astype(str)

                                if TOKENIZE:
                                    # FINAL TOKENIZATION THE DATA
                                    CLEANED_DATA_TOKENIZED = hero.tokenize(CLEANED_DATA['CLEANED CONTENT'])
                                    CLEANED_DATA_TOKENIZED = CLEANED_DATA_TOKENIZED.to_frame().astype(str)
                            except Exception as ex:
                                st.error(ex)

                if EXTEND_STOPWORD:
                    if FINALISE:
                        if TOKENIZE:
                            FINALISED_DATA_LIST = [
                                (DATA, 'Raw Data', 'raw_ascii_data.csv'),
                                (CLEANED_DATA, 'Cleaned Data', 'cleaned_data.csv'),
                                (CLEANED_DATA_TOKENIZED, 'Cleaned Tokenized Data', 'tokenized.csv')
                            ]
                        else:
                            FINALISED_DATA_LIST = [
                                (DATA, 'Raw Data', 'raw_ascii_data.csv'),
                                (CLEANED_DATA, 'Cleaned Data', 'cleaned_data.csv')
                            ]
                else:
                    if TOKENIZE:
                        FINALISED_DATA_LIST = [
                            (DATA, 'Raw Data', 'raw_ascii_data.csv'),
                            (CLEANED_DATA, 'Cleaned Data', 'cleaned_data.csv'),
                            (CLEANED_DATA_TOKENIZED, 'Cleaned Tokenized Data', 'tokenized.csv')
                        ]
                    else:
                        FINALISED_DATA_LIST = [
                            (DATA, 'Raw Data', 'raw_ascii_data.csv'),
                            (CLEANED_DATA, 'Cleaned Data', 'cleaned_data.csv')
                        ]

# -------------------------------------------------------------------------------------------------------------------- #
# +                                        VISUALISE THE DATA: CLEANED DATA                                          + #
# -------------------------------------------------------------------------------------------------------------------- #
                if VERBOSE:
                    if CLEAN:
                        if CLEAN_MODE == 'Simple':
                            st.markdown('## Cleaned DataFrame')
                            printDataFrame(data=CLEANED_DATA, verbose_level=VERBOSITY,
                                           advanced=ADVANCED_ANALYSIS)
                            if TOKENIZE:
                                st.markdown('## Cleaned Tokenized DataFrame')
                                printDataFrame(data=CLEANED_DATA_TOKENIZED, verbose_level=VERBOSITY,
                                               advanced=ADVANCED_ANALYSIS)
                        elif CLEAN_MODE == 'Complex':
                            if EXTEND_STOPWORD:
                                if FINALISE:
                                    st.markdown('## Cleaned DataFrame')
                                    printDataFrame(data=CLEANED_DATA, verbose_level=VERBOSITY,
                                                   advanced=ADVANCED_ANALYSIS)
                                    if TOKENIZE:
                                        st.markdown('## Cleaned Tokenized DataFrame')
                                        printDataFrame(data=CLEANED_DATA_TOKENIZED, verbose_level=VERBOSITY,
                                                       advanced=ADVANCED_ANALYSIS)
                            else:
                                st.markdown('## Cleaned DataFrame')
                                printDataFrame(data=CLEANED_DATA, verbose_level=VERBOSITY, advanced=ADVANCED_ANALYSIS)
                                if TOKENIZE:
                                    st.markdown('## Cleaned Tokenized DataFrame')
                                    printDataFrame(data=CLEANED_DATA_TOKENIZED, verbose_level=VERBOSITY,
                                                   advanced=ADVANCED_ANALYSIS)
                    else:
                        st.markdown('## Raw DataFrame')
                        printDataFrame(data=DATA, extract_from=DATA_COLUMN, verbose_level=VERBOSITY,
                                       advanced=ADVANCED_ANALYSIS)

# -------------------------------------------------------------------------------------------------------------------- #
# +                                                   SAVE THE DATA                                                  + #
# -------------------------------------------------------------------------------------------------------------------- #
                if SAVE:
                    if EXTEND_STOPWORD:
                        if FINALISE:
                            try:
                                st.markdown('## Download Data')
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
                        try:
                            st.markdown('## Download Data')
                            for data in FINALISED_DATA_LIST:
                                if not data[0].empty:
                                    st.markdown(f'### {data[1]}\n'
                                                f'Download data from [downloads/{data[2]}]'
                                                f'(downloads/{data[2]})')
                                    data[0].to_csv(str(DOWNLOAD_PATH / f'{data[2]}'), index=False)
                        except KeyError:
                            st.error('Warning: Your Data as not processed properly. Try again.')
                        except Exception as ex:
                            st.error(f'Error: Unknown Fatal Error -> {ex}')
            else:
                st.error('Error: No Files Uploaded.')
        else:
            st.error('Error: No Files Uploaded.')
