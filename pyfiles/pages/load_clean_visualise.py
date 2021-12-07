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
import re
import nltk
import numpy as np
import pandas as pd
import pycountry
import streamlit as st
import texthero as hero
from config import load_clean_visualise as lcv
from config import dtm, STREAMLIT_STATIC_PATH, DOWNLOAD_PATH

from texthero import stopwords
from collections import Counter
from texthero import preprocessing
import plotly.express as px
from utils import csp_downloaders
from utils.helper import readFile, lemmatizeText, downloadCorpora, printDataFrame
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder


# -------------------------------------------------------------------------------------------------------------------- #
# |                                                  INITIAL SETUP                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
# # DOWNLOAD THE NECESSARY CORPUS
downloadCorpora('words')


# -------------------------------------------------------------------------------------------------------------------- #
# |                                              MAIN APP FUNCTIONALITY                                              | #
# -------------------------------------------------------------------------------------------------------------------- #
def app():
    """
    Main function that will be called when the app is run and this module is called
    """

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                    INIT                                                          | #
# -------------------------------------------------------------------------------------------------------------------- #
    # CREATE THE DOWNLOAD PATH WHERE FILES CREATED WILL BE STORED AND AVAILABLE FOR DOWNLOADING
    if not DOWNLOAD_PATH.is_dir():
        DOWNLOAD_PATH.mkdir()

    st.title('Load, Clean and Visualise Data')
    st.markdown('## Init\n'
                'This module is used for the visualisation and cleaning of data used for NLP Analysis on news articles.'
                ' Since this process uses the `nltk` stopword corpus, you need to download the corpus onto your '
                'system. The corpus will automatically download onto your device when you run the app. Please ensure '
                'that sufficient storage (~1 GB) of storage space is free and that you are connected to the Internet '
                'to ensure that the corpus can be successfully downloaded. If the download fails, rerun the app again '
                'and ensure that your device has sufficient space and is connected to the Internet.\n\n '
                'For the cleaning process, all non-ASCII characters will be removed, and all non-English text '
                'will be removed. Multi-language support has not been implemented into this module as of yet.\n\n')

    st.markdown('## Processing Mode\n\n'
                'Choose the type of processing you want to apply to your dataset. You may choose between the three '
                'processes: **Cleaning**, **Modification (Country Extraction)** and **Query**.')
    lcv['ANALYSIS_MODE'] = st.selectbox('Choose Data Processing Mode', ('Data Cleaning', 'Data Modification',
                                                                        'Data Query'))
    st.info(f'**{lcv["ANALYSIS_MODE"]}** Mode Selected!')

    st.markdown('## Upload Data\n'
                'Due to limitations imposed by the file uploader widget, only files smaller than 200 MB can be loaded '
                'with the widget. To circumvent this limitation, you may choose to '
                'rerun the app with the tag `--server.maxUploadSize=[SIZE_IN_MB_HERE]` appended behind the '
                '`streamlit run app.py` command and define the maximum size of file you can upload '
                'onto Streamlit (replace `SIZE_IN_MB_HERE` with an integer value above). Do note that this option '
                'is only available for users who run the app using the app\'s source code or through Docker. '
                'For Docker, you will need to append the tag above behind the Docker Image name when running the `run` '
                'command, e.g. `docker run asdfghjklxl/news:latest --server.maxUploadSize=1028`; if you do not use '
                'the tag, the app will run with a default maximum upload size of 200 MB.\n\n'
                'Alternatively, you may use the Large File option to pull your dataset from any one of the three '
                'supported Cloud Service Providers into the app.\n\n'
                'After selecting the size of your file, select the file format you wish to upload. You are warned '
                'that if you fail to define the correct file format you wish to upload, the app will not let you '
                'upload your file (if you are using the Small File option, only the defined file format will be '
                'accepted by the File Uploader widget) and may result in errors (for Large File option).\n\n')
    lcv['FILE'] = st.selectbox('Select the Size of File to Load', ('Small File(s)', 'Large File(s)'))
    lcv['MODE'] = st.selectbox('Define the Data Input Format', ('CSV', 'XLSX'))


# -------------------------------------------------------------------------------------------------------------------- #
# |                                                 FILE UPLOADING                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    if lcv['FILE'] == 'Small File(s)':
        st.markdown('### Upload File\n')
        lcv['DATA_PATH'] = st.file_uploader(f'Load {lcv["MODE"]} File', type=[lcv["MODE"]])
        if lcv['DATA_PATH'] is not None:
            lcv['DATA'] = readFile(lcv['DATA_PATH'], lcv["MODE"])
            if not lcv['DATA'].empty:
                lcv['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored', list(lcv['DATA'].columns),
                                                  help='Note that if you select Data Modification, this field is '
                                                       'rendered invalid and the entire DataFrame will be used .')
                st.success(f'Data Loaded from **{lcv["DATA_COLUMN"]}**!')
        else:
            # RESET
            lcv['DATA'] = pd.DataFrame()

    elif lcv['FILE'] == 'Large File(s)':
        st.info(f'File Format Selected: **{lcv["MODE"]}**')
        lcv['CSP'] = st.selectbox('CSP', ('Select a CSP', 'Azure', 'Amazon', 'Google'))

        if lcv['CSP'] == 'Azure':
            azure = csp_downloaders.AzureDownloader()
            if azure.SUCCESSFUL:
                try:
                    azure.downloadBlob()
                    lcv['DATA'] = readFile(azure.AZURE_DOWNLOAD_PATH, lcv["MODE"])
                except Exception as ex:
                    lcv['DATA'] = pd.DataFrame()
                    st.error(f'Error: {ex}. Try again.')

            if not lcv['DATA'].empty and lcv['MOD_MODE'] != 'Inplace Data Modification':
                lcv['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored', list(lcv['DATA'].columns))
                st.success(f'Data Loaded from {lcv["DATA_COLUMN"]}!')

        elif lcv['CSP'] == 'Amazon':
            aws = csp_downloaders.AWSDownloader()
            if aws.SUCCESSFUL:
                try:
                    aws.downloadFile()
                    lcv['DATA'] = readFile(aws.AWS_FILE_NAME, lcv["MODE"])
                except Exception as ex:
                    lcv['DATA'] = pd.DataFrame()
                    st.error(f'Error: {ex}. Try again.')

            if not lcv['DATA'].empty and lcv['MOD_MODE'] != 'Inplace Data Modification':
                lcv['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored', list(lcv['DATA'].columns))
                st.success(f'Data Loaded from {lcv["DATA_COLUMN"]}!')

        elif lcv['CSP'] == 'Google':
            gcs = csp_downloaders.GoogleDownloader()
            if gcs.SUCCESSFUL:
                try:
                    gcs.downloadBlob()
                    lcv['DATA'] = readFile(gcs.GOOGLE_DESTINATION_FILE_NAME, lcv["MODE"])
                except Exception as ex:
                    lcv['DATA'] = pd.DataFrame()
                    st.error(f'Error: {ex}. Try again.')

            if not lcv['DATA'].empty:
                lcv['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored', list(lcv['DATA'].columns))
                st.success(f'Data Loaded from {lcv["DATA_COLUMN"]}!')

    # FLAGS
    if lcv['ANALYSIS_MODE'] == 'Data Cleaning':
        st.markdown('## Flags\n'
                    'Note that there is an size limit **(50 MB)** for the DataFrames that are printed to screen. If '
                    'you get an error telling you that the DataFrame size is too large to proceed, kindly lower the '
                    'number of data points you wish to visualise or download the file and visualise it through Excel '
                    'or any other DataFrame visualising Python packages. There is no definitive way to increase the '
                    'size of the DataFrame that can be printed out due to the inherent limitation on the size of the '
                    'packets sent over to and from the Streamlit server.')
        lcv['SAVE'] = st.checkbox('Save Outputs?', help='Note: Only Simple and Complex Cleaning modes will produce any '
                                                        'saved outputs. If None mode is chosen, you will not be able '
                                                        'to download the outputs as it is assumed that you already '
                                                        'possess that.\n\n'
                                                        'Additionally, due to the possibility of files with the same '
                                                        'file name and content being downloaded again, a unique file '
                                                        'identifier is tacked onto the filename.')
        lcv['VERBOSE'] = st.checkbox('Display Outputs?')
        if lcv['VERBOSE']:
            lcv['VERBOSITY'] = st.slider('Data Points To Print',
                                         key='Data points to display?',
                                         min_value=0,
                                         max_value=1000,
                                         value=20,
                                         help='Select 0 to display all Data Points')
            lcv['ADVANCED_ANALYSIS'] = st.checkbox('Display Advanced DataFrame Statistics?',
                                                   help='This option will analyse your DataFrame and display advanced '
                                                        'statistics on it. Note that this will require some time and '
                                                        'processing power to complete. Deselect this option if this '
                                                        'if you do not require it.')
        lcv['CLEAN_MODE'] = st.selectbox('Select Preprocessing Pipelines', ('None', 'Simple', 'Complex'),
                                         help='None mode will return the raw data passed into the app. This mode is '
                                              'recommended for purely visualising data without modifying it in any '
                                              'way.\n\n'
                                              'Simple mode will retain sentence structure and context, while removing '
                                              'any wrongly encoded characters.\n\n'
                                              'Complex mode will remove stopwords and lemmatize words; sentence '
                                              'structure and context may be destroyed in this process. Numbers will '
                                              'all be removed.\n\n'
                                              'Files created from simple cleaning may be used for Summarization in NLP '
                                              'Toolkit while files created from complex cleaning may be used for '
                                              'Document-Term Matrix Creation in Document-Term Matrix. Note that for '
                                              'most NLP Processes, the complex cleaning process is recommended.')
        lcv['TOKENIZE'] = st.checkbox('Tokenize Data?', value=True, help='This option is enabled by default. Deselect '
                                                                         'this option if you do not want to tokenize '
                                                                         'your data.')
        if lcv['CLEAN_MODE'] == 'Complex':
            lcv['EXTEND_STOPWORD'] = st.checkbox('Extend List of Stopwords?',
                                                 help='Select this option to extend the list of stopwords that will '
                                                      'be used to clean your data.')
            if lcv['EXTEND_STOPWORD']:
                lcv['STOPWORD_LIST'] = str()
                lcv['STOPWORD_LIST'] = st.text_area('Extended List of Stopwords',
                                                    value='Key in a list of stopwords that you wish to append to the '
                                                          'existing list of stopwords. Delimit your words by commas, '
                                                          'e.g. this, is, a, sample, list ...')
                if len(lcv['STOPWORD_LIST']) != 0:
                    st.info('Alert: Words detected.')
                else:
                    st.info('Alert: No Words detected')

    elif lcv['ANALYSIS_MODE'] == 'Data Modification':
        st.markdown('This module will allow you to modify the data passed in by performing certain elementary '
                    'analysis on the data. So far, we have implemented the ability to extract the countries '
                    'mentioned in your data and to plot out the Data Points on a World Map and the ability to '
                    'modify a single value of the inputted DataFrame in place.')
        st.markdown('## Data Modification Mode')
        lcv['MOD_MODE'] = st.selectbox('Choose Mode', ('Country Extraction', 'Inplace Data Modification'))
        if lcv['MOD_MODE'] == 'Country Extraction':
            st.markdown('## Flags\n'
                        'Note that there is an size limit **(50 MB)** for the DataFrames that are printed to screen. '
                        'If you get an error telling you that the DataFrame size is too large to proceed, kindly lower '
                        'the number of data points you wish to visualise or download the file and visualise it through '
                        'Excel or any other DataFrame visualising Python packages. There is no definitive way to '
                        'increase the size of the DataFrame that can be printed out due to the inherent limitation on '
                        'the size of the packets sent over to and from the Streamlit server.')
            lcv['SAVE'] = st.checkbox('Save Outputs?',
                                      help='Note: Only Simple and Complex Cleaning modes will produce any '
                                           'saved outputs. If None mode is chosen, you will not be able '
                                           'to download the outputs as it is assumed that you already '
                                           'possess that.\n\n'
                                           'Additionally, due to the possibility of files with the same '
                                           'file name and content being downloaded again, a unique file '
                                           'identifier is tacked onto the filename.')
            lcv['VERBOSE'] = st.checkbox('Display Outputs?')
            if lcv['VERBOSE']:
                lcv['VERBOSITY'] = st.slider('Data Points To Print',
                                             key='Data points to display?',
                                             min_value=0,
                                             max_value=1000,
                                             value=20,
                                             help='Select 0 to display all Data Points')
                lcv['ADVANCED_ANALYSIS'] = st.checkbox('Display Advanced DataFrame Statistics?',
                                                       help='This option will analyse your DataFrame and display '
                                                            'advanced statistics on it. Note that this will require '
                                                            'some time and processing power to complete. Deselect this '
                                                            'option if this if you do not require it.')
            if lcv['VERBOSE']:
                lcv['WORLD_MAP'] = st.checkbox('Generate a World Map Representation of the Countries Mentioned?',
                                               value=True)
        elif lcv['MOD_MODE'] == 'Inplace Data Modification':
            lcv['FIXED_KEY'] = st.checkbox('Use Fixed Key for Editing Table?')
            lcv['HEIGHT'] = st.number_input('Height of Table', min_value=100, max_value=800, value=400)

    elif lcv['ANALYSIS_MODE'] == 'Data Query':
        st.markdown('## Flags\n'
                    'Note that there is an size limit **(50 MB)** for the DataFrames that are printed to screen. If '
                    'you get an error telling you that the DataFrame size is too large to proceed, kindly lower the '
                    'number of data points you wish to visualise or download the file and visualise it through Excel '
                    'or any other DataFrame visualising Python packages. There is no definitive way to increase the '
                    'size of the DataFrame that can be printed out due to the inherent limitation on the size of the '
                    'packets sent over to and from the Streamlit server.')
        lcv['SAVE'] = st.checkbox('Save Outputs?', help='Note: Only Simple and Complex Cleaning modes will produce any '
                                                        'saved outputs. If None mode is chosen, you will not be able '
                                                        'to download the outputs as it is assumed that you already '
                                                        'possess that.\n\n'
                                                        'Additionally, due to the possibility of files with the same '
                                                        'file name and content being downloaded again, a unique file '
                                                        'identifier is tacked onto the filename.')
        lcv['VERBOSE'] = st.checkbox('Display Outputs?')
        if lcv['VERBOSE']:
            lcv['VERBOSITY'] = st.slider('Data Points To Print',
                                         key='Data points to display?',
                                         min_value=0,
                                         max_value=1000,
                                         value=20,
                                         help='Select 0 to display all Data Points')
            lcv['ADVANCED_ANALYSIS'] = st.checkbox('Display Advanced DataFrame Statistics?',
                                                   help='This option will analyse your DataFrame and display advanced '
                                                        'statistics on it. Note that this will require some time and '
                                                        'processing power to complete. Deselect this option if this '
                                                        'if you do not require it.')
        lcv['MATCH'] = st.checkbox('Query Must Match Exactly?', help='Select this option if you want your query string/'
                                                                     'condition to match exactly with your data.')
        lcv['QUERY_MODE'] = st.radio('Choose Method of Query', ('Single Query', 'Multiple Queries'),
                                     help='Note: For Single Query and Multiple Queries, the DataFrame values are '
                                          'converted to strings, hence all queries will be based on substring '
                                          'searching.')

        if lcv['QUERY_MODE'] == 'Single Query':
            lcv['QUERY'] = st.text_input('Key in words/numbers/characters/symbols to Query')
            if len(lcv['QUERY']) != 0:
                lcv['QUERY_SUCCESS'] = True
                st.info('Alert: Words detected.')
            else:
                lcv['QUERY_SUCCESS'] = False
                st.info('Alert: No Words detected')

        elif lcv['QUERY_MODE'] == 'Multiple Queries':
            lcv['QUERY'] = st.text_area('Key in words/numbers/characters/symbols to Query',
                                        help='Note: Delimit your list by commas')
            lcv['QUERY'] = [word.strip() for word in lcv['QUERY'].split(sep=',')]
            lcv['QUERY'] = '|'.join(map(re.escape, lcv['QUERY']))
            if len(lcv['QUERY']) != 0:
                lcv['QUERY_SUCCESS'] = True
                st.info('Alert: Words detected.')
            else:
                lcv['QUERY_SUCCESS'] = False
                st.info('Alert: No Words detected')


# -------------------------------------------------------------------------------------------------------------------- #
# |                                                DATA CLEANING                                                     | #
# -------------------------------------------------------------------------------------------------------------------- #
    if lcv['ANALYSIS_MODE'] == 'Data Cleaning':
        st.markdown('---')
        st.markdown('## Data Cleaning and Visualisation\n'
                    'Ensure that you have successfully uploaded the required data and selected the correct column '
                    'containing your data before clicking on the "Begin Analysis" button. The status of your file '
                    'upload is displayed below for your reference.')
        if lcv['FILE'] == 'Small File(s)':
            if lcv['DATA_PATH']:
                st.info('File loaded.')
            else:
                st.warning('File has not been loaded.')
        elif lcv['FILE'] == 'Large File(s)':
            if not lcv['DATA'].empty:
                st.info('File loaded.')
            else:
                st.warning('File has not been loaded.')

        if st.button('Begin Analysis', key='runner'):
            # RESET STATE
            lcv['CLEANED_DATA'] = pd.DataFrame()
            lcv['CLEANED_DATA_TOKENIZED'] = pd.DataFrame()

            if (lcv['FILE'] == 'Small File(s)' and lcv['DATA_PATH']) or \
                    (lcv['FILE'] == 'Large File(s)' and not lcv['DATA'].empty):
                if not lcv['DATA'].empty:
                    try:
                        lcv['DATA'] = lcv['DATA'].astype(str)
                        lcv['DATA'][lcv['DATA_COLUMN']] = lcv['DATA'][lcv['DATA_COLUMN']].str.encode('ascii', 'ignore')\
                            .str.decode('ascii')
                        lcv['DATA'] = pd.DataFrame(data=lcv['DATA'])
                        lcv['DATA'] = lcv['DATA'].dropna()
                    except Exception as ex:
                        st.error(f'Error: {ex}')

                    if lcv['CLEAN_MODE'] == 'None':
                        # DO NOTHING
                        lcv['DATA'] = lcv['DATA'][[lcv['DATA_COLUMN']]]
                        lcv['CLEANED_DATA_TOKENIZED'] = hero.tokenize(lcv['DATA'][lcv['DATA_COLUMN']])
                        lcv['CLEANED_DATA_TOKENIZED'] = lcv['CLEANED_DATA_TOKENIZED'].to_frame().astype(str)

                    elif lcv['CLEAN_MODE'] == 'Simple':
                        try:
                            lcv['CLEANED_DATA'] = lcv['DATA'][[lcv['DATA_COLUMN']]]

                            # PREPROCESSING AND CLEANING
                            lcv['CLEANED_DATA']['CLEANED CONTENT'] = hero.clean(lcv['CLEANED_DATA'][lcv['DATA_COLUMN']],
                                                                                lcv['SIMPLE_PIPELINE'])
                            lcv['CLEANED_DATA']['CLEANED CONTENT'].replace('', np.nan, inplace=True)
                            lcv['CLEANED_DATA'].dropna(inplace=True, subset=['CLEANED CONTENT'])

                            lcv['CLEANED_DATA'] = lcv['CLEANED_DATA'].astype(str)

                            if lcv['TOKENIZE']:
                                lcv['CLEANED_DATA_TOKENIZED'] = hero.tokenize(lcv['CLEANED_DATA']['CLEANED CONTENT'])
                                lcv['CLEANED_DATA_TOKENIZED'] = lcv['CLEANED_DATA_TOKENIZED'].to_frame().astype(str)
                        except Exception as ex:
                            st.error(ex)

                    elif lcv['CLEAN_MODE'] == 'Complex':
                        # CHECK IF NEW STOPWORDS WERE ADDED TO THE LIST
                        if lcv['EXTEND_STOPWORD']:
                            try:
                                if len(lcv['STOPWORD_LIST']) != 0:
                                    lcv['STOPWORD_LIST'] = set(word.strip().lower() for word in lcv['STOPWORD_LIST'].
                                                               split(sep=','))
                                    st.info(f'Stopwords accepted: {[word for word in lcv["STOPWORD_LIST"]]}!')
                                    lcv['STOPWORD_LIST'] = stopwords.DEFAULT.union(lcv['STOPWORD_LIST'])
                                    lcv['FINALISE'] = True
                                else:
                                    lcv['FINALISE'] = False
                                    lcv['STOPWORD_LIST'] = stopwords.DEFAULT
                                    raise ValueError('Length of Stopword List is 0. Try again.')
                            except Exception as ex:
                                st.error(f'Error: {ex}')

                        # IF NO NEW STOPWORDS WERE DEFINED, THEN USE DEFAULT
                        else:
                            lcv['STOPWORD_LIST'] = stopwords.DEFAULT
                            st.info('Alert: Default set of stopwords will be used.')
                            lcv['FINALISE'] = True

                        # NO ELSE CONDITION AS ELSE CONDITION IS EXPLICITLY SPECIFIED IN THE PREVIOUS EXCEPTION/ERROR
                        if lcv['FINALISE']:
                            try:
                                lcv['CLEANED_DATA'] = lcv['DATA'][[lcv['DATA_COLUMN']]]

                                # PREPROCESSING AND CLEANING
                                lcv['CLEANED_DATA']['CLEANED CONTENT'] = hero.clean(lcv['CLEANED_DATA']
                                                                                    [lcv['DATA_COLUMN']],
                                                                                    lcv['PIPELINE'])
                                lcv['CLEANED_DATA']['CLEANED CONTENT'] = hero.remove_digits(lcv['CLEANED_DATA']
                                                                                            ['CLEANED CONTENT'],
                                                                                            only_blocks=False)
                                lcv['CLEANED_DATA']['CLEANED CONTENT'] = hero.remove_stopwords(lcv['CLEANED_DATA']
                                                                                               ['CLEANED CONTENT'],
                                                                                               lcv["STOPWORD_LIST"])

                                lcv['CLEANED_DATA_TOKENIZED'] = hero.tokenize(lcv['CLEANED_DATA']['CLEANED CONTENT'])
                                lcv['CLEANED_DATA_TOKENIZED'] = lcv['CLEANED_DATA_TOKENIZED'].apply(lemmatizeText)

                                # ACCEPT ONLY ENGLISH WORDS
                                fin_list = [[word for word in text if word.lower() in lcv['ENGLISH_WORDS'] or not
                                            word.isalpha()] for text in lcv['CLEANED_DATA_TOKENIZED']]

                                # UPDATE TOKENS
                                lcv['CLEANED_DATA']['CLEANED CONTENT'] = [' '.join(text) for text in fin_list]
                                lcv['CLEANED_DATA_TOKENIZED'].update([str(text) for text in fin_list])
                                lcv['CLEANED_DATA_TOKENIZED'] = lcv['CLEANED_DATA_TOKENIZED'].to_frame().astype(str)
                                lcv['CLEANED_DATA']['CLEANED CONTENT'].replace('', np.nan, inplace=True)
                                lcv['CLEANED_DATA'].dropna(subset=['CLEANED CONTENT'], inplace=True)
                                lcv['CLEANED_DATA'] = lcv['CLEANED_DATA'].astype(str)
                            except Exception as ex:
                                st.error(ex)

                    if lcv['CLEAN_MODE'] == 'None':
                        if lcv['TOKENIZE']:
                            lcv['FINALISED_DATA_LIST'] = [(lcv['CLEANED_DATA_TOKENIZED'], 'Tokenized Data',
                                                           'tokenized.csv')]
                    elif lcv['CLEAN_MODE'] == 'Simple':
                        if lcv['TOKENIZE']:
                            lcv['FINALISED_DATA_LIST'] = [
                                (lcv['DATA'], 'Raw Data', 'raw_ascii_data.csv'),
                                (lcv['CLEANED_DATA'], 'Cleaned Data', 'cleaned_data.csv'),
                                (lcv['CLEANED_DATA_TOKENIZED'], 'Cleaned Tokenized Data', 'tokenized.csv')
                            ]
                        else:
                            lcv['FINALISED_DATA_LIST'] = [
                                (lcv['DATA'], 'Raw Data', 'raw_ascii_data.csv'),
                                (lcv['CLEANED_DATA'], 'Cleaned Data', 'cleaned_data.csv')
                            ]
                    elif lcv['CLEAN_MODE'] == 'Complex':
                        if lcv['TOKENIZE']:
                            lcv['FINALISED_DATA_LIST'] = [
                                (lcv['DATA'], 'Raw Data', 'raw_ascii_data.csv'),
                                (lcv['CLEANED_DATA'], 'Cleaned Data', 'cleaned_data.csv'),
                                (lcv['CLEANED_DATA_TOKENIZED'], 'Cleaned Tokenized Data', 'tokenized.csv')
                            ]
                        else:
                            lcv['FINALISED_DATA_LIST'] = [
                                (lcv['DATA'], 'Raw Data', 'raw_ascii_data.csv'),
                                (lcv['CLEANED_DATA'], 'Cleaned Data', 'cleaned_data.csv')
                            ]

                    if lcv['VERBOSE']:
                        if lcv['CLEAN_MODE'] == 'None':
                            st.markdown('## Raw DataFrame')
                            printDataFrame(data=lcv['DATA'], verbose_level=lcv['VERBOSITY'],
                                           advanced=lcv['ADVANCED_ANALYSIS'])
                            if lcv['TOKENIZE']:
                                st.markdown('## Tokenized DataFrame')
                                printDataFrame(data=lcv['CLEANED_DATA_TOKENIZED'], verbose_level=lcv['VERBOSITY'],
                                               advanced=lcv['ADVANCED_ANALYSIS'])
                        elif lcv['CLEAN_MODE'] == 'Simple':
                            st.markdown('## Cleaned DataFrame')
                            printDataFrame(data=lcv['CLEANED_DATA'], verbose_level=lcv['VERBOSITY'],
                                           advanced=lcv['ADVANCED_ANALYSIS'])
                            if lcv['TOKENIZE']:
                                st.markdown('## Cleaned Tokenized DataFrame')
                                printDataFrame(data=lcv['CLEANED_DATA_TOKENIZED'], verbose_level=lcv['VERBOSITY'],
                                               advanced=lcv['ADVANCED_ANALYSIS'])
                        elif lcv['CLEAN_MODE'] == 'Complex':
                            if lcv['EXTEND_STOPWORD']:
                                if lcv['FINALISE']:
                                    st.markdown('## Cleaned DataFrame')
                                    printDataFrame(data=lcv['CLEANED_DATA'], verbose_level=lcv['VERBOSITY'],
                                                   advanced=lcv['ADVANCED_ANALYSIS'])
                                    if lcv['TOKENIZE']:
                                        st.markdown('## Cleaned Tokenized DataFrame')
                                        printDataFrame(data=lcv['CLEANED_DATA_TOKENIZED'],
                                                       verbose_level=lcv['VERBOSITY'],
                                                       advanced=lcv['ADVANCED_ANALYSIS'])
                            else:
                                st.markdown('## Cleaned DataFrame')
                                printDataFrame(data=lcv['CLEANED_DATA'], verbose_level=lcv['VERBOSITY'],
                                               advanced=lcv['ADVANCED_ANALYSIS'])
                                if lcv['TOKENIZE']:
                                    st.markdown('## Cleaned Tokenized DataFrame')
                                    printDataFrame(data=lcv['CLEANED_DATA_TOKENIZED'], verbose_level=lcv['VERBOSITY'],
                                                   advanced=lcv['ADVANCED_ANALYSIS'])

                    if lcv['SAVE']:
                        if lcv['CLEAN_MODE'] == 'None':
                            if lcv['TOKENIZE']:
                                try:
                                    st.markdown('---')
                                    st.markdown('## Download Data')
                                    for data in lcv['FINALISED_DATA_LIST']:
                                        if not data[0].empty:
                                            st.markdown(f'### {data[1]}\n'
                                                        f'Download data from [downloads/{data[2]}]'
                                                        f'(downloads/id{lcv["FC"]}_{data[2]})')
                                            data[0].to_csv(str(DOWNLOAD_PATH / f'id{lcv["FC"]}_{data[2]}'), index=False)
                                            lcv["FC"] += 1
                                except KeyError:
                                    st.error('Warning: Your data was not processed properly. Try again.')
                                except Exception as ex:
                                    st.error(f'Error: Unknown Fatal Error -> {ex}')
                            else:
                                st.info('None mode chosen, no file downloads are provided.')

                        elif lcv['CLEAN_MODE'] == 'Simple':
                            try:
                                st.markdown('---')
                                st.markdown('## Download Data')
                                for data in lcv['FINALISED_DATA_LIST']:
                                    if not data[0].empty:
                                        st.markdown(f'### {data[1]}\n'
                                                    f'Download data from [downloads/{data[2]}]'
                                                    f'(downloads/id{lcv["FC"]}_{data[2]})')
                                        data[0].to_csv(str(DOWNLOAD_PATH / f'id{lcv["FC"]}_{data[2]}'), index=False)
                                        lcv["FC"] += 1
                            except KeyError:
                                st.error('Warning: Your data was not processed properly. Try again.')
                            except Exception as ex:
                                st.error(f'Error: Unknown Fatal Error -> {ex}')

                        elif lcv['CLEAN_MODE'] == 'Complex':
                            if lcv['EXTEND_STOPWORD']:
                                if lcv['FINALISE']:
                                    try:
                                        st.markdown('---')
                                        st.markdown('## Download Data')
                                        for data in lcv['FINALISED_DATA_LIST']:
                                            if not data[0].empty:
                                                st.markdown(f'### {data[1]}\n'
                                                            f'Download data from [downloads/{data[2]}]'
                                                            f'(downloads/id{lcv["FC"]}_{data[2]})')
                                                data[0].to_csv(str(DOWNLOAD_PATH / f'id{lcv["FC"]}_{data[2]}'),
                                                               index=False)
                                                lcv["FC"] += 1
                                    except KeyError:
                                        st.error('Warning: Your data was not processed properly. Try again.')
                                    except Exception as ex:
                                        st.error(f'Error: Unknown Fatal Error -> {ex}')
                            else:
                                try:
                                    st.markdown('---')
                                    st.markdown('## Download Data')
                                    for data in lcv['FINALISED_DATA_LIST']:
                                        if not data[0].empty:
                                            st.markdown(f'### {data[1]}\n'
                                                        f'Download data from [downloads/{data[2]}]'
                                                        f'(downloads/id{lcv["FC"]}_{data[2]})')
                                            data[0].to_csv(str(DOWNLOAD_PATH / f'id{lcv["FC"]}_{data[2]}'), index=False)
                                            lcv["FC"] += 1
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
    elif lcv['ANALYSIS_MODE'] == 'Data Modification':
        if lcv['MOD_MODE'] == 'Country Extraction':
            st.markdown('---')
            st.markdown('## Country Extraction\n'
                        'This module will take in a DataFrame containing all documents meant for NLP Analysis and '
                        'return a new DataFrame whereby countries mentioned in the documents will be extracted. Users '
                        'can then choose to either output a set of countries mentioned in the document, or generate a '
                        'graphical representation of the frequency of country name occurrence within the set of '
                        'documents passed to it.')

            if lcv['FILE'] == 'Small File(s)':
                if lcv['DATA_PATH']:
                    st.info('File loaded.')
                else:
                    st.warning('File has not been loaded.')
            elif lcv['FILE'] == 'Large File(s)':
                if not lcv['DATA'].empty:
                    st.info('File loaded.')
                else:
                    st.warning('File has not been loaded.')

            if st.button('Begin Country Extraction', key='country'):
                lcv['GLOBE_DATA'] = pd.DataFrame()
                lcv['GLOBE_FIG'] = None

                if not lcv['DATA'].empty:
                    lcv['DATA'] = lcv['DATA'].astype(object)
                    lcv['DATA']['COUNTRIES'] = lcv['DATA'][lcv['DATA_COLUMN']].astype(str).apply(
                        lambda x: [country.name for country in pycountry.countries if country.name.lower() in
                                   x.lower()])
                    new_list = lcv['DATA']['COUNTRIES'].to_list()
                    temp = []
                    for ls in new_list:
                        temp.extend(ls)
                    zipped = list(zip(Counter(temp).keys(), Counter(temp).values()))

                    lcv['GLOBE_DATA'] = pd.DataFrame(data=zipped, index=range(len(zipped)),
                                                     columns=['country', 'count'])
                    lcv['GLOBE_FIG'] = px.scatter_geo(data_frame=lcv['GLOBE_DATA'], projection='natural earth',
                                                      color='country', locations='country', size='count',
                                                      hover_name='country', locationmode='country names',
                                                      title='Country Name Mention Frequency')

                    if lcv['VERBOSE']:
                        st.markdown('## Country Name Mention Frequency')
                        printDataFrame(data=lcv['GLOBE_DATA'], verbose_level=lcv['VERBOSITY'],
                                       advanced=lcv['ADVANCED_ANALYSIS'])
                        if lcv['WORLD_MAP']:
                            st.markdown('## World Map Representation')
                            st.plotly_chart(lcv['GLOBE_FIG'])

                    if lcv['SAVE']:
                        try:
                            st.markdown('---')
                            st.markdown('## Download Data')
                            st.markdown('### Country Data')
                            st.markdown(f'Download data from [downloads/globe_data.csv]'
                                        f'(downloads/globe_data_id{lcv["FC"]}.csv)')
                            lcv['DATA'].to_csv(str(DOWNLOAD_PATH / f'globe_data_id{lcv["FC"]}.csv'), index=False)
                            lcv["FC"] += 1

                            st.markdown('### Country Data (Concatenated)')
                            st.markdown(f'Download data from [downloads/globe_data_concat.csv]'
                                        f'(downloads/globe_data_concat_id{lcv["FC"]}.csv)')
                            lcv['GLOBE_DATA'].to_csv(str(DOWNLOAD_PATH / f'globe_data_concat_id{lcv["FC"]}.csv'),
                                                     index=False)
                            lcv["FC"] += 1

                            if lcv['WORLD_MAP']:
                                st.markdown('### World Map Representation')
                                st.markdown(f'Download data from [downloads/map.png]'
                                            f'(downloads/map_id{lcv["FC"]}.png)')
                                lcv['GLOBE_FIG'].write_image(str(DOWNLOAD_PATH / f'map_id{lcv["FC"]}.png'))
                                lcv["FC"] += 1
                        except ValueError:
                            st.warning('Error: Not connected to the Internet. Plot may not be generated properly. '
                                       'Connect to the Internet and try again.')
                        except Exception as ex:
                            st.error(f'Error: Unknown Fatal Error -> {ex}')
                else:
                    st.error('Error: No files loaded.')

        elif lcv['MOD_MODE'] == 'Inplace Data Modification':
            st.markdown('---')
            st.markdown('## Inplace Data Modification\n'
                        'This function uses the AgGrid module to create editable tables for your to edit your '
                        'DataFrame as you would with an Excel sheet.')

            if lcv['FILE'] == 'Small File(s)':
                if lcv['DATA_PATH']:
                    st.info('File loaded.')
                    gb = GridOptionsBuilder.from_dataframe(lcv['DATA'])
                    gb.configure_columns(lcv['DATA'].columns, editable=True)
                    go = gb.build()

                    if lcv['FIXED_KEY']:
                        ag = AgGrid(
                            lcv['DATA'],
                            gridOptions=go,
                            height=lcv['HEIGHT'],
                            fit_columns_on_grid_load=True,
                            key='data',
                            reload_data=False
                        )
                    else:
                        ag = AgGrid(
                            lcv['DATA'],
                            gridOptions=go,
                            height=lcv['HEIGHT'],
                            fit_columns_on_grid_load=True
                        )

                    if st.button('Generate Modified Data'):
                        st.markdown('### Modified Data')
                        st.markdown(f'Download data from [downloads/modified_data.csv]'
                                    f'(downloads/modified_data_id{lcv["FC"]}.csv)')
                        ag['data'].to_csv(str(DOWNLOAD_PATH / f'modified_data_id{lcv["FC"]}.csv'), index=False)
                        lcv["FC"] += 1
                else:
                    st.warning('File has not been loaded.')

            elif lcv['FILE'] == 'Large File(s)':
                if not lcv['DATA'].empty:
                    st.info('File loaded.')
                    gb = GridOptionsBuilder.from_dataframe(lcv['DATA'])
                    gb.configure_columns(lcv['DATA'].columns, editable=True)
                    go = gb.build()

                    if lcv['FIXED_KEY']:
                        ag = AgGrid(
                            lcv['DATA'],
                            gridOptions=go,
                            height=lcv['HEIGHT'],
                            fit_columns_on_grid_load=True,
                            key='data',
                            reload_data=False
                        )
                    else:
                        ag = AgGrid(
                            lcv['DATA'],
                            gridOptions=go,
                            height=lcv['HEIGHT'],
                            fit_columns_on_grid_load=True
                        )

                    if st.button('Generate Modified Data'):
                        if lcv['SAVE']:
                            st.markdown('### Modified Data')
                            st.markdown(f'Download data from [downloads/modified_data.csv]'
                                        f'(downloads/modified_data_id{lcv["FC"]}.csv)')
                            ag['data'].to_csv(str(DOWNLOAD_PATH / f'modified_data_id{lcv["FC"]}.csv'), index=False)
                            lcv["FC"] += 1
                else:
                    st.warning('File has not been loaded.')


# -------------------------------------------------------------------------------------------------------------------- #
# |                                                      QUERY                                                       | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif lcv['ANALYSIS_MODE'] == 'Data Query':
        st.markdown('---')
        st.markdown('## DataFrame Query\n'
                    'This module will allow you to query certain parts of your DataFrame to get documents which '
                    'fulfills your criteria. You may choose between querying just one column at a time (one '
                    'condition per query) or mulitple columns at a time (multiple conditions per query).')

        if lcv['FILE'] == 'Small File(s)':
            if lcv['DATA_PATH']:
                st.info('File loaded.')
            else:
                st.warning('File has not been loaded.')
        elif lcv['FILE'] == 'Large File(s)':
            if not lcv['DATA'].empty:
                st.info('File loaded.')
            else:
                st.warning('File has not been loaded.')

        if st.button('Query Data', key='query'):
            # reset query
            lcv['QUERY_DATA'] = pd.DataFrame()
            lcv['DATA'] = lcv['DATA'].astype(str)

            if not lcv['DATA'].empty:
                if lcv['QUERY_SUCCESS']:
                    try:
                        # make a copy of the original dataframe to avoid mutating it with .loc
                        temp = lcv['DATA'].copy()
                        lcv['QUERY_DATA'] = temp.loc[temp[lcv['DATA_COLUMN']].str.contains(lcv['QUERY'],
                                                                                           case=lcv['MATCH'])]
                    except Exception as ex:
                        st.error(f'Error: {ex}')
                    else:
                        st.success('Query Successful!')

                    if not lcv['QUERY_DATA'].empty:
                        if lcv['VERBOSE']:
                            st.markdown('## Query Results')
                            printDataFrame(data=lcv['QUERY_DATA'], verbose_level=lcv['VERBOSITY'],
                                           advanced=lcv['ADVANCED_ANALYSIS'])
                        if lcv['SAVE']:
                            st.markdown('---')
                            st.markdown('## Save Query')
                            st.markdown(f'Download data from [downloads/query.csv](downloads/query_id{lcv["FC"]}.csv)')
                            lcv['QUERY_DATA'].to_csv(str(DOWNLOAD_PATH / f'query_id{lcv["FC"]}.csv'), index=False)
                            lcv["FC"] += 1
                    else:
                        st.error('Query did not find matching keywords. Try again.')
                else:
                    st.error('Error: Query cannot be empty/empty strings. Try again.')
            else:
                st.error('Error: File not loaded properly. Try again.')
