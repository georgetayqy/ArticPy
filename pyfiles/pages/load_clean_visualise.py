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
import itertools
import pathlib
import re

import nltk
import numpy as np
import pandas as pd
import pycountry
import streamlit as st
import texthero as hero

from collections import Counter
from texthero import preprocessing
import plotly.express as px
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
TOKENIZE = True
EXTEND_STOPWORD = False
STOPWORD_LIST = str()
FIN_STOPWORD_LIST = []
FINALISE = False
ANALYSIS_MODE = 'Data Cleaning'
WORLD_MAP = True
GLOBE_DATA = None
GLOBE_FIG = None
MATCH = False
QUERY = None
QUERY_MODE = None
QUERY_DATA = pd.DataFrame()


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
        TOKENIZE, EXTEND_STOPWORD, STOPWORD_LIST, FIN_STOPWORD_LIST, FINALISE, ANALYSIS_MODE, WORLD_MAP, GLOBE_DATA, \
        GLOBE_FIG, QUERY_MODE, QUERY_DATA, MATCH

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
    st.markdown('## Processing Mode\n\n'
                'Choose the type of processing you want to apply to your dataset. You may choose between the three '
                'processes: Cleaning, Modification (Country Extraction) and Query.')
    ANALYSIS_MODE = st.selectbox('Choose Data Processing Mode', ('Data Cleaning', 'Data Modification', 'Data Query'))

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                 FILE UPLOADING                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    if FILE == 'Small File(s)':
        st.markdown('### Upload File:\n')
        DATA_PATH = st.file_uploader(f'Load {MODE} File', type=[MODE])
        if DATA_PATH is not None:
            DATA = readFile(DATA_PATH, MODE)
            if not DATA.empty:
                DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                st.success(f'Data Loaded from {DATA_COLUMN}!')
        else:
            # RESET
            DATA = pd.DataFrame()

    elif FILE == 'Large File(s)':
        st.info(f'File Format Selected: {MODE}')
        CSP = st.selectbox('CSP', ('Select a CSP', 'Azure', 'Amazon', 'Google'))

        if CSP == 'Azure':
            azure = csp_downloaders.AzureDownloader()
            if azure.SUCCESSFUL:
                try:
                    azure.downloadBlob()
                    DATA = readFile(azure.AZURE_DOWNLOAD_PATH, MODE)
                except Exception as ex:
                    DATA = pd.DataFrame()
                    st.error(f'Error: {ex}. Try again.')

            if not DATA.empty:
                DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                st.success(f'Data Loaded from {DATA_COLUMN}!')

        elif CSP == 'Amazon':
            aws = csp_downloaders.AWSDownloader()
            if aws.SUCCESSFUL:
                try:
                    aws.downloadFile()
                    DATA = readFile(aws.AWS_FILE_NAME, MODE)
                except Exception as ex:
                    DATA = pd.DataFrame()
                    st.error(f'Error: {ex}. Try again.')

            if not DATA.empty:
                DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                st.success(f'Data Loaded from {DATA_COLUMN}!')

        elif CSP == 'Google':
            gcs = csp_downloaders.GoogleDownloader()
            if gcs.SUCCESSFUL:
                try:
                    gcs.downloadBlob()
                    DATA = readFile(gcs.GOOGLE_DESTINATION_FILE_NAME, MODE)
                except Exception as ex:
                    DATA = pd.DataFrame()
                    st.error(f'Error: {ex}. Try again.')

            if not DATA.empty:
                DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                st.success(f'Data Loaded from {DATA_COLUMN}!')


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
    SAVE = st.checkbox('Save Outputs?', help='Note: Only Simple and Complex Cleaning modes will produce any saved '
                                             'outputs. If None mode is chosen, you will not be able to download '
                                             'the outputs as it is assumed that you already possess that.')
    VERBOSE = st.checkbox('Print out DataFrames?')

    if ANALYSIS_MODE == 'Data Cleaning':
        if VERBOSE:
            VERBOSITY = st.slider('Data Points To Print',
                                  key='Data points to display?',
                                  min_value=1,
                                  max_value=1000,
                                  value=20)
            ADVANCED_ANALYSIS = st.checkbox('Display Advanced DataFrame Statistics?',
                                            help='This option will analyse your DataFrame and display advanced '
                                                 'statistics on it. Note that this will require some time and '
                                                 'processing power to complete. Deselect this option if this if '
                                                 'you do not require ')
        CLEAN_MODE = st.selectbox('Select Preprocessing Pipelines', ('None', 'Simple', 'Complex'),
                                  help='None mode will return the raw data passed into the app. This mode is '
                                       'recommended for purely visualising data without modifying it in any way.\n\n'
                                       'Simple mode will retain sentence structure and context, while removing any '
                                       'wrongly encoded characters.\n\n'
                                       'Complex mode will remove stopwords and lemmatize words; sentence structure '
                                       'and context may be destroyed in this process.\n\n'
                                       'Files created from simple cleaning may be used for Summarization in NLP '
                                       'Toolkit while files created from complex cleaning may be used for '
                                       'Document-Term Matrix Creation in Document-Term Matrix. Note that for most '
                                       'NLP Processes, the complex cleaning process is recommended.')
        TOKENIZE = st.checkbox('Tokenize Data?', value=True, help='This option is enabled by default. Deselect '
                                                                  'this option if you do not want to tokenize '
                                                                  'your data.')
        if CLEAN_MODE == 'Complex':
            EXTEND_STOPWORD = st.checkbox('Extend List of Stopwords?',
                                          help='Select this option to extend the list of stopwords that will '
                                               'be used to clean your data.')
            if EXTEND_STOPWORD:
                STOPWORD_LIST = str()
                STOPWORD_LIST = st.text_area('Extended List of Stopwords',
                                             value='Key in a list of stopwords that you wish to append to the '
                                                   'existing list of stopwords. Delimit your words by commas, '
                                                   'e.g. this, is, a, sample, list ...')
                if len(STOPWORD_LIST) != 0:
                    st.info('Alert: Words detected.')
                else:
                    st.info('Alert: No Words detected')

    elif ANALYSIS_MODE == 'Data Modification':
        st.markdown('This module will allow you to modify the data passed in by performing certain elementary '
                    'analysis on the data. So far, we have implemented the ability to extract the countries '
                    'mentioned in your data and to plot out the Data Points on a World Map.')
        if VERBOSE:
            VERBOSITY = st.slider('Data Points To Print',
                                  key='Data points to display?',
                                  min_value=1,
                                  max_value=1000,
                                  value=20)
            ADVANCED_ANALYSIS = st.checkbox('Display Advanced DataFrame Statistics?',
                                            help='This option will analyse your DataFrame and display advanced '
                                                 'statistics on it. Note that this will require some time and '
                                                 'processing power to complete. Deselect this option if this '
                                                 'functionality is not required.')
            WORLD_MAP = st.checkbox('Generate a World Map Representation of the Countries Mentioned?', value=True)

    elif ANALYSIS_MODE == 'Data Query':
        MATCH = st.checkbox('Query Must Match Exactly?', help='Select this option if you want your query string/'
                                                              'condition to match exactly with your data.')
        QUERY_MODE = st.radio('Choose Method of Query', ('Single Query', 'Multiple Queries'),
                              help='Note: For Single Query and Multiple Queries, the DataFrame values are converted to '
                                   'strings, hence all queries will be based on substring searching.')

        if QUERY_MODE == 'Single Query':
            QUERY = st.text_input('Key in word(s)/numbers/character/symbol to Query')
            if len(QUERY) != 0:
                st.info('Alert: Words detected.')
            else:
                st.info('Alert: No Words detected')

        elif QUERY_MODE == 'Multiple Queries':
            QUERY = st.text_area('Key in word(s)/numbers/character/symbol to Query',
                                 help='Note: Delimit your list by commas')
            QUERY = [word.strip() for word in QUERY.split(sep=',')]
            QUERY = '|'.join(map(re.escape, QUERY))
            if len(QUERY) != 0:
                st.info('Alert: Words detected.')
            else:
                st.info('Alert: No Words detected')


# -------------------------------------------------------------------------------------------------------------------- #
# |                                                DATA CLEANING                                                     | #
# -------------------------------------------------------------------------------------------------------------------- #
    if ANALYSIS_MODE == 'Data Cleaning':
        st.markdown('## Data Cleaning and Visualisation\n'
                    'Ensure that you have successfully uploaded the required data and selected the correct column '
                    'containing your data before clicking on the "Begin Analysis" button. The status of your file '
                    'upload is displayed below for your reference.')
        if FILE == 'Small File(s)':
            if DATA_PATH:
                st.info('File loaded.')
            else:
                st.warning('File has not been loaded.')
        elif FILE == 'Large File(s)':
            if not DATA.empty:
                st.info('File loaded.')
            else:
                st.warning('File has not been loaded.')

        if st.button('Begin Analysis', key='runner'):
            # RESET STATE
            CLEANED_DATA = pd.DataFrame()
            CLEANED_DATA_TOKENIZED = pd.DataFrame()

            if (FILE == 'Small File(s)' and DATA_PATH) or (FILE == 'Large File(s)' and not DATA.empty):
                if not DATA.empty:
                    try:
                        DATA = DATA.astype(str)
                        DATA[DATA_COLUMN] = DATA[DATA_COLUMN].str.encode('ascii', 'ignore').str.decode('ascii')
                        DATA = pd.DataFrame(data=DATA)
                        DATA = DATA.dropna()
                    except Exception as ex:
                        st.error(f'Error: {ex}')

                    if CLEAN_MODE == 'None':
                        # DO NOTHING
                        DATA = DATA[[DATA_COLUMN]]

                    elif CLEAN_MODE == 'Simple':
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

                    elif CLEAN_MODE == 'Complex':
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

                    if CLEAN_MODE == 'None':
                        FINALISED_DATA_LIST = [(DATA, 'Raw Data', 'raw_ascii_data.csv')]
                    elif CLEAN_MODE == 'Simple' or CLEAN_MODE == 'Complex':
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

                    if VERBOSE:
                        if CLEAN_MODE == 'None':
                            st.markdown('## Raw DataFrame')
                            printDataFrame(data=DATA, verbose_level=VERBOSITY, advanced=ADVANCED_ANALYSIS)
                        elif CLEAN_MODE == 'Simple':
                            st.markdown('## Cleaned DataFrame')
                            printDataFrame(data=CLEANED_DATA, verbose_level=VERBOSITY, advanced=ADVANCED_ANALYSIS)
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

                    if SAVE:
                        if CLEAN_MODE == 'None':
                            st.info('None mode chosen, no file downloads are provided.')

                        elif CLEAN_MODE == 'Simple':
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

                        elif CLEAN_MODE == 'Complex':
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


# -------------------------------------------------------------------------------------------------------------------- #
# |                                               COUNTRY EXTRACTION                                                 | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif ANALYSIS_MODE == 'Data Modification':
        st.markdown('## Country Extraction\n'
                    'This module will take in a DataFrame containing all documents meant for NLP Analysis and return '
                    'a new DataFrame whereby countries mentioned in the documents will be extracted. Users can then '
                    'choose to either output a set of countries mentioned in the document, or generate a graphical '
                    'representation of the frequency of country name occurrence within the set of documents passed '
                    'to it.')

        if FILE == 'Small File(s)':
            if DATA_PATH:
                st.info('File loaded.')
            else:
                st.warning('File has not been loaded.')
        elif FILE == 'Large File(s)':
            if not DATA.empty:
                st.info('File loaded.')
            else:
                st.warning('File has not been loaded.')

        if st.button('Begin Country Extraction', key='country'):
            if not DATA.empty:
                DATA = DATA.astype(object)
                DATA['COUNTRIES'] = DATA[DATA_COLUMN].astype(str).apply(lambda x: [country.name for country in
                                                                                   pycountry.countries if
                                                                                   country.name.lower() in x.lower()])
                new_list = DATA['COUNTRIES'].to_list()
                temp = []
                for ls in new_list:
                    temp.extend(ls)
                zipped = list(zip(Counter(temp).keys(), Counter(temp).values()))

                GLOBE_DATA = pd.DataFrame(data=zipped, index=range(len(zipped)), columns=['country', 'count'])
                GLOBE_FIG = px.scatter_geo(data_frame=GLOBE_DATA, projection='natural earth', color='country',
                                           locations='country', size='count', hover_name='country',
                                           locationmode='country names', title='Country Name Mention Frequency')

                if VERBOSE:
                    st.markdown('## Country Name Mention Frequency')
                    printDataFrame(data=GLOBE_DATA, verbose_level=VERBOSITY,
                                   advanced=ADVANCED_ANALYSIS)
                    if WORLD_MAP:
                        st.markdown('## World Map Representation')
                        st.plotly_chart(GLOBE_FIG)

                if SAVE:
                    try:
                        st.markdown('## Download Data')
                        st.markdown('### Country Data')
                        st.markdown(f'Download data from [downloads/globe_data.csv]'
                                    f'(downloads/globe_data.csv)')
                        DATA.to_csv(str(DOWNLOAD_PATH / 'globe_data.csv'), index=False)

                        st.markdown('### Country Data (Concatenated)')
                        st.markdown(f'Download data from [downloads/globe_data_concat.csv]'
                                    f'(downloads/globe_data_concat.csv)')
                        GLOBE_DATA.to_csv(str(DOWNLOAD_PATH / 'globe_data_concat.csv'), index=False)

                        if WORLD_MAP:
                            st.markdown('### World Map Representation')
                            st.markdown(f'Download data from [downloads/map.png]'
                                        f'(downloads/map.png)')
                            GLOBE_FIG.write_image(str(DOWNLOAD_PATH / 'map.png'))
                    except Exception as ex:
                        st.error(f'Error: Unknown Fatal Error -> {ex}')
            else:
                st.error('Warning: Your data was not processed properly. Try again.')


# -------------------------------------------------------------------------------------------------------------------- #
# |                                                      QUERY                                                       | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif ANALYSIS_MODE == 'Data Query':
        st.markdown('## DataFrame Query\n'
                    'This module will allow you to query certain parts of your DataFrame to get documents which '
                    'fulfills your criteria. You may choose between querying just one column at a time (one '
                    'condition per query) or mulitple columns at a time (multiple conditions per query).')

        if FILE == 'Small File(s)':
            if DATA_PATH:
                st.info('File loaded.')
            else:
                st.warning('File has not been loaded.')
        elif FILE == 'Large File(s)':
            if not DATA.empty:
                st.info('File loaded.')
            else:
                st.warning('File has not been loaded.')

        if st.button('Query Data', key='query'):
            if not DATA.empty:
                try:
                    DATA = DATA.astype(str)
                    QUERY_DATA = DATA.loc[DATA[DATA_COLUMN].str.contains(QUERY, case=MATCH)]
                except Exception as ex:
                    st.error(f'Error: {ex}')
                else:
                    st.success('Query Successful!')

                if VERBOSE:
                    if not QUERY_DATA.empty:
                        st.markdown('## Query Results')
                        printDataFrame(data=QUERY_DATA, verbose_level=VERBOSITY, advanced=ADVANCED_ANALYSIS)
                    else:
                        st.error('Error: Query was not Successful. Try again.')

                if SAVE:
                    if not QUERY_DATA.empty:
                        st.markdown('## Save Query')
                        st.markdown('Download data from [downloads/query.csv](downloads/query.csv)')
                        QUERY_DATA.to_csv(str(DOWNLOAD_PATH / 'query.csv'), index=False)
                    else:
                        st.error('Error: Query was not Successful. Try again.')
            else:
                st.error('Error: File not loaded properly. Try again.')
