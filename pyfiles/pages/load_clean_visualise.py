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
import re
import numpy as np
import pandas as pd
import pycountry
import streamlit as st
import texthero as hero
import platform
import pathlib
import os
from config import load_clean_visualise as lcv

from streamlit_tags import st_tags
from texthero import stopwords
from collections import Counter
from plotly.io import to_image
import plotly.express as px
from utils import csp_downloaders
from utils.helper import readFile, lemmatizeText, printDataFrame, prettyDownload
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder


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
    st.title('Load, Clean and Visualise Data')
    with st.expander('Module Description'):
        st.markdown('This module is used for the visualisation and cleaning of data used for NLP Analysis on news '
                    'articles. Since this process uses the `nltk` stopword corpus, the corpus will automatically '
                    'download onto your device when you run the app. Please ensure that sufficient storage (~1 GB) '
                    'of storage space is free and that you are connected to the Internet to ensure that the corpus '
                    'can be successfully downloaded. If the download fails, rerun the app again and ensure that your '
                    'device has sufficient space and is connected to the Internet.\n\n '
                    'For the cleaning process, all non-ASCII characters will be removed, and all non-English text '
                    'will be removed. Multi-language support has not been implemented into this module as of yet.\n\n')
        st.markdown('Before proceeding, you will need to download the corpus needed to process your data. To do so, '
                    'click on the "Begin Download" button below. Please ensure that you have at least 3 GB of free '
                    'disk space available so that you are able to download the corpus onto your system and that your '
                    'device is connected to the Internet.')

        # CHECK IF THE DATA HAS BEEN DOWNLOADED
        path_flag = False
        files_flag = False

        if platform.system() == 'Windows':
            possible_fp = [pathlib.Path.joinpath(pathlib.Path.home(), 'AppData', 'Roaming', 'nltk_data'),
                           pathlib.Path.joinpath(pathlib.Path.cwd(), 'nltk_data'),
                           pathlib.Path.joinpath(pathlib.Path.cwd(), 'lib', 'nltk_data'),
                           pathlib.Path(r'C:\nltk_data'),
                           pathlib.Path(r'D:\nltk_data'),
                           pathlib.Path(r'E:\nltk_data')]
            for paths in possible_fp:
                if paths.is_dir():
                    path_flag = True
                    if any(paths.iterdir()):
                        files_flag = True

            if path_flag and files_flag:
                st.info('NTLK Data Detected')
            else:
                st.warning('NLTK Data Not Detected')

        elif platform.system() == 'Linux':
            possible_fp = [pathlib.Path.joinpath(pathlib.Path.home(), 'share', 'nltk_data'),
                           pathlib.Path.joinpath(pathlib.Path.home(), 'local', 'share', 'nltk_data'),
                           pathlib.Path.joinpath(pathlib.Path.home(), 'lib', 'nltk_data'),
                           pathlib.Path.joinpath(pathlib.Path.home(), 'local', 'lib', 'nltk_data'),
                           pathlib.Path.joinpath(pathlib.Path.cwd(), 'nltk_data'),
                           pathlib.Path.joinpath(pathlib.Path.cwd(), 'lib', 'nltk_data')]
            for paths in possible_fp:
                if paths.is_dir():
                    path_flag = True
                    if any(paths.iterdir()):
                        files_flag = True

            if path_flag and files_flag:
                st.info('NTLK Data Detected')
            else:
                st.warning('NLTK Data Not Detected')

        elif platform.system() == 'Darwin':
            possible_fp = [pathlib.Path.joinpath(pathlib.Path.home(), 'share', 'nltk_data'),
                           pathlib.Path.joinpath(pathlib.Path.home(), 'local', 'share', 'nltk_data'),
                           pathlib.Path.joinpath(pathlib.Path.home(), 'lib', 'nltk_data'),
                           pathlib.Path.joinpath(pathlib.Path.home(), 'local', 'lib', 'nltk_data'),
                           pathlib.Path.joinpath(pathlib.Path.home(), 'nltk_data'),
                           pathlib.Path.joinpath(pathlib.Path.cwd(), 'nltk_data'),
                           pathlib.Path.joinpath(pathlib.Path.cwd(), 'lib', 'nltk_data')]
            for paths in possible_fp:
                if paths.is_dir():
                    path_flag = True
                    if any(paths.iterdir()):
                        files_flag = True

            if path_flag and files_flag:
                st.info('NTLK Data Detected')
            else:
                st.warning('NLTK Data Not Detected')

        if st.button('Begin Download', key='download-model'):
            os.system('python -m nltk.downloader all')

    st.markdown('## Processing Mode\n\n'
                'Choose the type of processing you want to apply to your dataset. You may choose between the three '
                'processes: **Cleaning**, **Modification (Country Extraction)** and **Query**.')
    lcv['ANALYSIS_MODE'] = st.selectbox('Choose Data Processing Mode', ('Data Cleaning', 'Data Modification',
                                                                        'Data Query'),
                                        help='**Data Cleaning**: \tThis mode cleans the data for further processing\n\n'
                                             '**Data Modification**: \tThis mode allows users to modify their data by '
                                             'adding in new information or to change existing information.\n\n'
                                             '**Data Query**: \tThis mode allows users to query their data for '
                                             'specific keywords of interest.',
                                        key='lcv-mode')
    st.info(f'**{lcv["ANALYSIS_MODE"]}** Mode Selected')

    st.markdown('## Upload Data\n')
    col1, col1_ = st.columns(2)
    lcv['FILE'] = col1.selectbox('Origin of Data File', ('Local', 'Online'),
                                 help='Choose "Local" if you wish to upload a file from your machine or choose '
                                      '"Online" if you wish to pull a file from any one of the supported Cloud '
                                      'Service Providers.',
                                 key='lcv-origin')
    lcv['MODE'] = col1_.selectbox('Define the Data Input Format', ('CSV', 'XLSX', 'PKL', 'JSON'),
                                  key='lcv-mode')


# -------------------------------------------------------------------------------------------------------------------- #
# |                                                 FILE UPLOADING                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    if lcv['FILE'] == 'Local':
        lcv['DATA_PATH'] = st.file_uploader(f'Load {lcv["MODE"]} File', type=[lcv["MODE"]], key='lcv-fp')
        if lcv['DATA_PATH'] is not None:
            lcv['DATA'] = readFile(lcv['DATA_PATH'], lcv["MODE"])
            if not lcv['DATA'].empty:
                lcv['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored', list(lcv['DATA'].columns),
                                                  help='Note that if you select Data Modification, this field is '
                                                       'rendered invalid and the entire DataFrame will be used .',
                                                  key='lcv-dc')
                st.success(f'Data Loaded from **{lcv["DATA_COLUMN"]}**!')
        else:
            # RESET
            st.warning('Warning: Your Dataset file is not loaded.')
            lcv['DATA'] = pd.DataFrame()

    elif lcv['FILE'] == 'Online':
        st.info(f'File Format Selected: **{lcv["MODE"]}**')
        lcv['CSP'] = st.selectbox('CSP', ('Select a CSP', 'Azure', 'Amazon', 'Google'), key='lcv-csp')

        if lcv['CSP'] == 'Azure':
            azure = csp_downloaders.AzureDownloader()
            if azure.SUCCESSFUL:
                try:
                    azure.downloadBlob()
                    lcv['DATA'] = readFile(azure.AZURE_DOWNLOAD_PATH, lcv["MODE"])
                except Exception as ex:
                    lcv['DATA'] = pd.DataFrame()
                    st.error(f'Error: {ex}. Try again.')
            else:
                st.error('Error establishing connection with Azure and pulling data...')

            if not lcv['DATA'].empty and lcv['MOD_MODE'] != 'Inplace Data Modification':
                lcv['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored', list(lcv['DATA'].columns),
                                                  key='lcv-az')
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
            else:
                st.error('Error establishing connection with Azure and pulling data...')

            if not lcv['DATA'].empty and lcv['MOD_MODE'] != 'Inplace Data Modification':
                lcv['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored', list(lcv['DATA'].columns),
                                                  key='lcv-aws')
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
                lcv['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored', list(lcv['DATA'].columns),
                                                  key='lcv-gcs')
                st.success(f'Data Loaded from {lcv["DATA_COLUMN"]}!')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                      FLAGS                                                       | #
# -------------------------------------------------------------------------------------------------------------------- #
    if lcv['ANALYSIS_MODE'] == 'Data Cleaning':
        st.markdown('## Options\n')
        lcv['SAVE'] = st.checkbox('Save Outputs?', help='Note: Only Simple and Complex Cleaning modes will produce any '
                                                        'saved outputs. If None mode is chosen, you will not be able '
                                                        'to download the outputs as it is assumed that you already '
                                                        'possess that.\n\n',
                                  key='lcv-save-cleaning')
        if lcv['SAVE']:
            if st.checkbox('Override Output Format?', key='override-cleaning'):
                lcv['OVERRIDE_FORMAT'] = st.selectbox('Overridden Output Format',
                                                      ('CSV', 'XLSX', 'PKL', 'JSON'),
                                                      key='lcv-override-cleaning')
                if lcv['OVERRIDE_FORMAT'] == lcv['MODE']:
                    st.warning('Warning: Overridden Format is the same as Input Format')
            else:
                lcv['OVERRIDE_FORMAT'] = None

        lcv['VERBOSE'] = st.checkbox('Display Outputs?',
                                     help='Note that there is an size limit for the DataFrames that are '
                                          'printed to screen. If you get an error telling you that the DataFrame size '
                                          'is too large to proceed, kindly lower the number of data points you wish '
                                          'to visualise or increase the maximum size of items to print to screen '
                                          'through the maxMessageSize setting in the Streamlit config file.',
                                     key='lcv-vb-cleaning')
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
                                                        'if you do not require it.',
                                                   key='lcv-advanced-cleaning')
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
                                              'most NLP Processes, the complex cleaning process is recommended.',
                                         key='lcv-pipeline-cleaning')
        lcv['TOKENIZE'] = st.checkbox('Tokenize Data?', value=True, help='This option is enabled by default. Deselect '
                                                                         'this option if you do not want to tokenize '
                                                                         'your data.',
                                      key='lcv-tokenize-cleaning')

        if lcv['CLEAN_MODE'] == 'Complex':
            lcv['EXTEND_STOPWORD'] = st.checkbox('Extend List of Stopwords?',
                                                 help='Select this option to extend the list of stopwords that will '
                                                      'be used to clean your data.',
                                                 key='lcv-extension-cleaning')
            if lcv['EXTEND_STOPWORD']:
                lcv['STOPWORD_LIST'] = st_tags(label='**Keyword List**',
                                               text='Press Enter to extend list...',
                                               maxtags=9999999,
                                               key='lcv-extended-cleaning')
                if len(lcv['STOPWORD_LIST']) == 0:
                    st.info('**Alert**: No Words detected')

    elif lcv['ANALYSIS_MODE'] == 'Data Modification':
        st.markdown('## Data Modification Mode')
        lcv['MOD_MODE'] = st.selectbox('Choose Mode', ('Country Extraction', 'Inplace Data Modification'),
                                       help='This module will allow you to modify the data passed in by performing '
                                            'certain elementary analysis on the data. So far, we have implemented the '
                                            'ability to extract the countries mentioned in your data and to plot out '
                                            'the Data Points on a World Map and the ability to modify a single value '
                                            'of the inputted DataFrame in place.',
                                       key='lcv-mode-modification')
        if lcv['MOD_MODE'] == 'Country Extraction':
            st.markdown('## Options\n')
            lcv['SAVE'] = st.checkbox('Save Outputs?',
                                      help='Note: Only Simple and Complex Cleaning modes will produce any '
                                           'saved outputs. If None mode is chosen, you will not be able '
                                           'to download the outputs as it is assumed that you already '
                                           'possess that.\n\n',
                                      key='lcv-save-modification')
            if lcv['SAVE']:
                if st.checkbox('Override Output Format?', key='override-modification'):
                    lcv['OVERRIDE_FORMAT'] = st.selectbox('Overridden Output Format',
                                                          ('CSV', 'XLSX', 'PKL', 'JSON'),
                                                          key='override-format-modification')
                    if lcv['OVERRIDE_FORMAT'] == lcv['MODE']:
                        st.warning('Warning: Overridden Format is the same as Input Format')
                else:
                    lcv['OVERRIDE_FORMAT'] = None

            lcv['VERBOSE'] = st.checkbox('Display Outputs?',
                                         help='Note that there is an size limit for the DataFrames that are '
                                              'printed to screen. If you get an error telling you that the DataFrame '
                                              'size is too large to proceed, kindly lower the number of data points '
                                              'you wish to visualise or increase the maximum size of items to print '
                                              'to screen through the maxMessageSize setting in the Streamlit config '
                                              'file.',
                                         key='vb-modification')
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
                                                            'option if this if you do not require it.',
                                                       key='vb-advanced')
            if lcv['VERBOSE']:
                lcv['WORLD_MAP'] = st.checkbox('Generate a World Map Representation of the Countries Mentioned?',
                                               value=True, key='world-map')
        elif lcv['MOD_MODE'] == 'Inplace Data Modification':
            lcv['FIXED_KEY'] = st.checkbox('Use Fixed Key for Editing Table?', key='lcv-fixedkeys-modification')
            lcv['HEIGHT'] = st.number_input('Height of Table', min_value=100, max_value=800, value=400,
                                            key='lcv-height')

    elif lcv['ANALYSIS_MODE'] == 'Data Query':
        st.markdown('## Options\n')
        lcv['SAVE'] = st.checkbox('Save Outputs?', help='Note: Only Simple and Complex Cleaning modes will produce any '
                                                        'saved outputs. If None mode is chosen, you will not be able '
                                                        'to download the outputs as it is assumed that you already '
                                                        'possess that.\n\n',
                                  key='lcv-save-query')
        if lcv['SAVE']:
            if st.checkbox('Override Output Format?', key='lcv-override-query'):
                lcv['OVERRIDE_FORMAT'] = st.selectbox('Overridden Output Format',
                                                      ('CSV', 'XLSX', 'PKL', 'JSON'),
                                                      key='lcv-override-format-query')
                if lcv['OVERRIDE_FORMAT'] == lcv['MODE']:
                    st.warning('Warning: Overridden Format is the same as Input Format')
            else:
                lcv['OVERRIDE_FORMAT'] = None

        lcv['VERBOSE'] = st.checkbox('Display Outputs?',
                                     help='Note that there is an size limit for the DataFrames that are '
                                          'printed to screen. If you get an error telling you that the DataFrame size '
                                          'is too large to proceed, kindly lower the number of data points you wish '
                                          'to visualise or increase the maximum size of items to print to screen '
                                          'through the maxMessageSize setting in the Streamlit config file.',
                                     key='vb-query')
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
                                                        'if you do not require it.',
                                                   key='lcv-advanced-query')
        lcv['MATCH'] = st.checkbox('Query Must Match Exactly?', help='Select this option if you want your query string/'
                                                                     'condition to match exactly with your data.',
                                   key='lcv-match-query')

        lcv['QUERY'] = st_tags(label='**Query**',
                               text='Press Enter to extend list...',
                               maxtags=9999999,
                               key='query_input')
        if len(lcv['QUERY']) == 0:
            lcv['QUERY_SUCCESS'] = False
            st.info('Alert: No Words detected')
        elif len(lcv['QUERY']) == 1:
            lcv['QUERY_SUCCESS'] = True
            st.info('Alert: Words detected.')
        else:
            lcv['QUERY'] = '|'.join(map(re.escape, lcv['QUERY']))
            lcv['QUERY_SUCCESS'] = True
            st.info('Alert: Words detected.')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                DATA CLEANING                                                     | #
# -------------------------------------------------------------------------------------------------------------------- #
    if lcv['ANALYSIS_MODE'] == 'Data Cleaning':
        st.markdown('---')
        st.markdown('## Data Cleaning and Visualisation\n'
                    'Ensure that you have successfully uploaded the required data and selected the correct column '
                    'containing your data before clicking on the "Begin Analysis" button. The status of your file '
                    'upload is displayed below for your reference.')
        if lcv['FILE'] == 'Local':
            if lcv['DATA_PATH']:
                st.info('File loaded.')
            else:
                st.warning('File has not been loaded.')
        elif lcv['FILE'] == 'Online':
            if not lcv['DATA'].empty:
                st.info('File loaded.')
            else:
                st.warning('File has not been loaded.')

        if st.button('Begin Analysis', key='clean'):
            # RESET STATE
            lcv['CLEANED_DATA'] = pd.DataFrame()
            lcv['CLEANED_DATA_TOKENIZED'] = pd.DataFrame()

            if (lcv['FILE'] == 'Local' and lcv['DATA_PATH']) or \
                    (lcv['FILE'] == 'Online' and not lcv['DATA'].empty):
                if not lcv['DATA'].empty:
                    try:
                        lcv['DATA'].dropna(inplace=True)
                        lcv['DATA'][lcv['DATA_COLUMN']] = lcv['DATA'][lcv['DATA_COLUMN']].apply(lambda x: x.encode(
                            'ascii', 'ignore').decode('ascii'))
                        lcv['DATA'] = pd.DataFrame(data=lcv['DATA'])

                    except Exception as ex:
                        st.error(f'Error: {ex}')

                    if lcv['CLEAN_MODE'] == 'None':
                        # DO NOTHING
                        lcv['DATA'] = lcv['DATA'][[lcv['DATA_COLUMN']]]

                        if lcv['TOKENIZE']:
                            lcv['CLEANED_DATA_TOKENIZED'] = hero.tokenize(lcv['DATA'][lcv['DATA_COLUMN']])
                            lcv['CLEANED_DATA_TOKENIZED'] = lcv['CLEANED_DATA_TOKENIZED'].to_frame()

                    elif lcv['CLEAN_MODE'] == 'Simple':
                        try:
                            lcv['CLEANED_DATA'] = lcv['DATA'][[lcv['DATA_COLUMN']]]

                            # PREPROCESSING AND CLEANING
                            lcv['CLEANED_DATA']['CLEANED CONTENT'] = hero.clean(lcv['CLEANED_DATA'][lcv['DATA_COLUMN']],
                                                                                lcv['SIMPLE_PIPELINE'])
                            lcv['CLEANED_DATA']['CLEANED CONTENT'].replace('', np.nan, inplace=True)
                            lcv['CLEANED_DATA'].dropna(inplace=True, subset=['CLEANED CONTENT'])

                            if lcv['TOKENIZE']:
                                lcv['CLEANED_DATA_TOKENIZED'] = hero.tokenize(lcv['CLEANED_DATA']['CLEANED CONTENT'])
                                lcv['CLEANED_DATA_TOKENIZED'] = lcv['CLEANED_DATA_TOKENIZED'].to_frame()
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
                                lcv['CLEANED_DATA_TOKENIZED'] = lcv['CLEANED_DATA_TOKENIZED'].to_frame()
                                lcv['CLEANED_DATA']['CLEANED CONTENT'].replace('', np.nan, inplace=True)
                                lcv['CLEANED_DATA'].dropna(subset=['CLEANED CONTENT'], inplace=True)
                            except Exception as ex:
                                st.error(ex)

                    if lcv['CLEAN_MODE'] == 'None':
                        if lcv['TOKENIZE']:
                            lcv['FINALISED_DATA_LIST'] = [(lcv['CLEANED_DATA_TOKENIZED'], 'Tokenized Data',
                                                           'tokenized', '.csv', False)]
                    elif lcv['CLEAN_MODE'] == 'Simple':
                        if lcv['TOKENIZE']:
                            lcv['FINALISED_DATA_LIST'] = [
                                (lcv['DATA'], 'Raw Data', 'raw_ascii_data', '.csv', False),
                                (lcv['CLEANED_DATA'], 'Cleaned Data', 'cleaned_data', '.csv', False),
                                (lcv['CLEANED_DATA_TOKENIZED'], 'Cleaned Tokenized Data', 'tokenized', '.csv', False)
                            ]
                        else:
                            lcv['FINALISED_DATA_LIST'] = [
                                (lcv['DATA'], 'Raw Data', 'raw_ascii_data', '.csv', False),
                                (lcv['CLEANED_DATA'], 'Cleaned Data', 'cleaned_data', '.csv', False)
                            ]
                    elif lcv['CLEAN_MODE'] == 'Complex':
                        if lcv['TOKENIZE']:
                            lcv['FINALISED_DATA_LIST'] = [
                                (lcv['DATA'], 'Raw Data', 'raw_ascii_data', '.csv', False),
                                (lcv['CLEANED_DATA'], 'Cleaned Data', 'cleaned_data', '.csv', False),
                                (lcv['CLEANED_DATA_TOKENIZED'], 'Cleaned Tokenized Data', 'tokenized', '.csv', False)
                            ]
                        else:
                            lcv['FINALISED_DATA_LIST'] = [
                                (lcv['DATA'], 'Raw Data', 'raw_ascii_data', '.csv', False),
                                (lcv['CLEANED_DATA'], 'Cleaned Data', 'cleaned_data', '.csv', False)
                            ]

                    if lcv['VERBOSE']:
                        if lcv['CLEAN_MODE'] == 'None':
                            st.markdown('### Raw DataFrame')
                            printDataFrame(data=lcv['DATA'], verbose_level=lcv['VERBOSITY'],
                                           advanced=lcv['ADVANCED_ANALYSIS'])
                            if lcv['TOKENIZE']:
                                st.markdown('### Tokenized DataFrame')
                                printDataFrame(data=lcv['CLEANED_DATA_TOKENIZED'], verbose_level=lcv['VERBOSITY'],
                                               advanced=lcv['ADVANCED_ANALYSIS'])
                        elif lcv['CLEAN_MODE'] == 'Simple':
                            st.markdown('### Cleaned DataFrame')
                            printDataFrame(data=lcv['CLEANED_DATA'], verbose_level=lcv['VERBOSITY'],
                                           advanced=lcv['ADVANCED_ANALYSIS'])
                            if lcv['TOKENIZE']:
                                st.markdown('### Cleaned Tokenized DataFrame')
                                printDataFrame(data=lcv['CLEANED_DATA_TOKENIZED'], verbose_level=lcv['VERBOSITY'],
                                               advanced=lcv['ADVANCED_ANALYSIS'])
                        elif lcv['CLEAN_MODE'] == 'Complex':
                            if lcv['EXTEND_STOPWORD']:
                                if lcv['FINALISE']:
                                    st.markdown('### Cleaned DataFrame')
                                    printDataFrame(data=lcv['CLEANED_DATA'], verbose_level=lcv['VERBOSITY'],
                                                   advanced=lcv['ADVANCED_ANALYSIS'])
                                    if lcv['TOKENIZE']:
                                        st.markdown('### Cleaned Tokenized DataFrame')
                                        printDataFrame(data=lcv['CLEANED_DATA_TOKENIZED'],
                                                       verbose_level=lcv['VERBOSITY'],
                                                       advanced=lcv['ADVANCED_ANALYSIS'])
                            else:
                                st.markdown('### Cleaned DataFrame')
                                printDataFrame(data=lcv['CLEANED_DATA'], verbose_level=lcv['VERBOSITY'],
                                               advanced=lcv['ADVANCED_ANALYSIS'])
                                if lcv['TOKENIZE']:
                                    st.markdown('### Cleaned Tokenized DataFrame')
                                    printDataFrame(data=lcv['CLEANED_DATA_TOKENIZED'], verbose_level=lcv['VERBOSITY'],
                                                   advanced=lcv['ADVANCED_ANALYSIS'])

                    if lcv['SAVE']:
                        if lcv['CLEAN_MODE'] == 'None':
                            if lcv['TOKENIZE']:
                                try:
                                    st.markdown('---')
                                    st.markdown('### Download Data')
                                    for index, data in enumerate(lcv['FINALISED_DATA_LIST']):
                                        if lcv['OVERRIDE_FORMAT'] is not None:
                                            st.markdown(prettyDownload(
                                                object_to_download=data[0],
                                                download_filename=f'{data[2]}.{lcv["OVERRIDE_FORMAT"].lower()}',
                                                button_text=f'Download {data[1]}',
                                                override_index=data[4],
                                                format_=lcv['OVERRIDE_FORMAT']),
                                                unsafe_allow_html=True)
                                        else:
                                            st.markdown(prettyDownload(
                                                object_to_download=data[0],
                                                download_filename=f'{data[2]}{data[3]}',
                                                button_text=f'Download {data[1]}',
                                                override_index=data[4],
                                                format_=lcv["MODE"]),
                                                unsafe_allow_html=True
                                            )
                                except KeyError:
                                    st.error('Warning: Your data was not processed properly. Try again.')
                                except Exception as ex:
                                    st.error(f'Error: Unknown Fatal Error -> {ex}')
                            else:
                                st.info('None mode chosen, no file downloads are provided.')

                        elif lcv['CLEAN_MODE'] == 'Simple':
                            try:
                                st.markdown('---')
                                st.markdown('### Download Data')
                                for index, data in enumerate(lcv['FINALISED_DATA_LIST']):
                                    if lcv['OVERRIDE_FORMAT'] is not None:
                                        st.markdown(prettyDownload(
                                            object_to_download=data[0],
                                            download_filename=f'{data[2]}.{lcv["OVERRIDE_FORMAT"].lower()}',
                                            button_text=f'Download {data[1]}',
                                            override_index=data[4],
                                            format_=lcv['OVERRIDE_FORMAT']),
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        st.markdown(prettyDownload(
                                            object_to_download=data[0],
                                            download_filename=f'{data[2]}{data[3]}',
                                            button_text=f'Download {data[1]}',
                                            override_index=data[4],
                                            format_=lcv["MODE"]),
                                            unsafe_allow_html=True
                                        )
                            except KeyError:
                                st.error('Warning: Your data was not processed properly. Try again.')
                            except Exception as ex:
                                st.error(f'Error: Unknown Fatal Error -> {ex}')

                        elif lcv['CLEAN_MODE'] == 'Complex':
                            st.markdown('---')
                            st.markdown('### Download Data')
                            if lcv['EXTEND_STOPWORD']:
                                if lcv['FINALISE']:
                                    try:
                                        for index, data in enumerate(lcv['FINALISED_DATA_LIST']):
                                            st.markdown('---')
                                            st.markdown('## Download Data')
                                            if lcv['OVERRIDE_FORMAT'] is not None:
                                                st.markdown(prettyDownload(
                                                    object_to_download=data[0],
                                                    download_filename=f'{data[2]}.{lcv["OVERRIDE_FORMAT"].lower()}',
                                                    button_text=f'Download {data[1]}',
                                                    override_index=data[4],
                                                    format_=lcv['OVERRIDE_FORMAT']),
                                                    unsafe_allow_html=True
                                                )
                                            else:
                                                st.markdown(prettyDownload(
                                                    object_to_download=data[0],
                                                    download_filename=f'{data[2]}{data[3]}',
                                                    button_text=f'Download {data[1]}',
                                                    override_index=data[4],
                                                    format_=lcv["MODE"]),
                                                    unsafe_allow_html=True
                                                )
                                    except KeyError:
                                        st.error('Warning: Your data was not processed properly. Try again.')
                                    except Exception as ex:
                                        st.error(f'Error: Unknown Fatal Error -> {ex}')
                            else:
                                if lcv['FINALISE']:
                                    try:
                                        for index, data in enumerate(lcv['FINALISED_DATA_LIST']):
                                            if lcv['OVERRIDE_FORMAT'] is not None:
                                                st.markdown(prettyDownload(
                                                    object_to_download=data[0],
                                                    download_filename=f'{data[2]}.{lcv["OVERRIDE_FORMAT"].lower()}',
                                                    button_text=f'Download {data[1]}',
                                                    override_index=data[4],
                                                    format_=lcv['OVERRIDE_FORMAT']),
                                                    unsafe_allow_html=True
                                                )
                                            else:
                                                st.markdown(prettyDownload(
                                                    object_to_download=data[0],
                                                    download_filename=f'{data[2]}{data[3]}',
                                                    button_text=f'Download {data[1]}',
                                                    override_index=data[4],
                                                    format_=lcv["MODE"]),
                                                    unsafe_allow_html=True
                                                )
                                    except KeyError:
                                        st.error('Warning: Your Data as not processed properly. Try again.')
                                    except Exception as ex:
                                        st.error(f'Error: Unknown Fatal Error -> {ex}')
                                else:
                                    st.error('Hmm... For some reason your data was not processed properly. Try again.')
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

            if lcv['FILE'] == 'Local':
                if lcv['DATA_PATH']:
                    st.info('File loaded.')
                else:
                    st.warning('File has not been loaded.')
            elif lcv['FILE'] == 'Online':
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
                        st.markdown('### Country Name Mention Frequency')
                        printDataFrame(data=lcv['GLOBE_DATA'], verbose_level=lcv['VERBOSITY'],
                                       advanced=lcv['ADVANCED_ANALYSIS'])
                        if lcv['WORLD_MAP']:
                            st.markdown('### World Map Representation')
                            st.plotly_chart(lcv['GLOBE_FIG'])

                    if lcv['SAVE']:
                        try:
                            st.markdown('---')
                            st.markdown('### Download Data')
                            if lcv['OVERRIDE_FORMAT'] is not None:
                                st.markdown(prettyDownload(
                                    object_to_download=lcv['DATA'],
                                    download_filename=f'globe_data.{lcv["OVERRIDE_FORMAT"].lower()}',
                                    button_text=f'Download Globe Data',
                                    override_index=False,
                                    format_=lcv['OVERRIDE_FORMAT']),
                                    unsafe_allow_html=True
                                )
                                st.markdown(prettyDownload(
                                    object_to_download=lcv['GLOBE_DATA'],
                                    download_filename=f'globe_data_concat.{lcv["OVERRIDE_FORMAT"].lower()}',
                                    button_text=f'Download Concatenated Globe Data',
                                    override_index=False,
                                    format_=lcv['OVERRIDE_FORMAT']),
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(prettyDownload(
                                    object_to_download=lcv['DATA'],
                                    download_filename=f'globe_data.{lcv["MODE"]}',
                                    button_text=f'Download Globe Data',
                                    override_index=False,
                                    format_=lcv["MODE"]),
                                    unsafe_allow_html=True
                                )
                                st.markdown(prettyDownload(
                                    object_to_download=lcv['GLOBE_DATA'],
                                    download_filename=f'globe_data_concat.csv',
                                    button_text=f'Download Concatenated Globe Data',
                                    override_index=False,
                                    format_=lcv["MODE"]),
                                    unsafe_allow_html=True
                                )
                            if lcv['WORLD_MAP']:
                                st.markdown(prettyDownload(
                                    object_to_download=lcv['GLOBE_FIG'],
                                    download_filename='map.png',
                                    button_text=f'Download Map Representation',
                                    override_index=False),
                                    unsafe_allow_html=True
                                )
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

            if lcv['FILE'] == 'Local':
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

                    if st.button('Generate Modified Data', key='modified_data'):
                        if lcv['OVERRIDE_FORMAT'] is not None:
                            st.markdown(prettyDownload(
                                object_to_download=ag['data'],
                                download_filename=f'modified_data.{lcv["OVERRIDE_FORMAT"].lower()}',
                                button_text=f'Download Modified Data',
                                override_index=False,
                                format_=lcv['OVERRIDE_FORMAT']),
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(prettyDownload(
                                object_to_download=ag['data'],
                                download_filename=f'modified_data.{lcv["MODE"].lower()}',
                                button_text=f'Download Modified Data',
                                override_index=False,
                                format_=lcv["MODE"]),
                                unsafe_allow_html=True
                            )
                else:
                    st.warning('File has not been loaded.')

            elif lcv['FILE'] == 'Online':
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

                    if st.button('Generate Modified Data', key='modify_other'):
                        if lcv['SAVE']:
                            st.markdown('### Modified Data')
                            if lcv['OVERRIDE_FORMAT'] is not None:
                                st.markdown(prettyDownload(
                                    object_to_download=ag['data'],
                                    download_filename=f'modified_data.{lcv["OVERRIDE_FORMAT"].lower()}',
                                    button_text=f'Download Modified Data',
                                    override_index=False,
                                    format_=lcv['OVERRIDE_FORMAT']),
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(prettyDownload(
                                    object_to_download=ag['data'],
                                    download_filename=f'modified_data.{lcv["MODE"].lower()}',
                                    button_text=f'Download Modified Data',
                                    override_index=False,
                                    format_=lcv["MODE"]),
                                    unsafe_allow_html=True
                                )
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

        if lcv['FILE'] == 'Local':
            if lcv['DATA_PATH']:
                st.info('File loaded.')
            else:
                st.warning('File has not been loaded.')
        elif lcv['FILE'] == 'Online':
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
                        # COPY DATAFRAME TO AVOID MUTATIONS WITH LOC
                        temp = lcv['DATA'].copy()
                        lcv['QUERY_DATA'] = temp.loc[temp[lcv['DATA_COLUMN']].str.contains('|'.join(lcv['QUERY']),
                                                                                           case=lcv['MATCH'])]
                    except Exception as ex:
                        st.error(f'Error: {ex}')
                    else:
                        st.success('Query Successful!')

                    if not lcv['QUERY_DATA'].empty:
                        if lcv['VERBOSE']:
                            st.markdown('### Query Results')
                            printDataFrame(data=lcv['QUERY_DATA'], verbose_level=lcv['VERBOSITY'],
                                           advanced=lcv['ADVANCED_ANALYSIS'])
                        if lcv['SAVE']:
                            st.markdown('---')
                            st.markdown('### Save Query')
                            if lcv['OVERRIDE_FORMAT'] is not None:
                                st.markdown(prettyDownload(object_to_download=lcv['QUERY_DATA'],
                                                           download_filename=f'query.{lcv["OVERRIDE_FORMAT"].lower()}',
                                                           button_text=f'Download Download Queried Data',
                                                           override_index=False,
                                                           format_=lcv['OVERRIDE_FORMAT']),
                                            unsafe_allow_html=True)
                            else:
                                st.markdown(prettyDownload(object_to_download=lcv['QUERY_DATA'],
                                                           download_filename=f'query.{lcv["MODE"].lower()}',
                                                           button_text=f'Download Queried Data',
                                                           override_index=False,
                                                           format_=lcv["MODE"]),
                                            unsafe_allow_html=True)
                    else:
                        st.error('Query did not find matching keywords. Try again.')
                else:
                    st.error('Error: Query cannot be empty/empty strings. Try again.')
            else:
                st.error('Error: File not loaded properly. Try again.')
