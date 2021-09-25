"""
This module is used for the main purpose of preparing and creating a Document-Term Matrix.

This module will use the sklearn feature extraction function to create the Document-Term Matrix.
"""

# -------------------------------------------------------------------------------------------------------------------- #
# |                                         IMPORT RELEVANT LIBRARIES                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
import pathlib

import openpyxl
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from utils import csp_downloaders
from utils.helper import readFile

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                      FLAGS                                                       | #
# -------------------------------------------------------------------------------------------------------------------- #
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
DOWNLOAD_PATH = (STREAMLIT_STATIC_PATH / "downloads")
DTM = pd.DataFrame()
pd.options.plotting.backend = 'plotly'


# -------------------------------------------------------------------------------------------------------------------- #
# |                                          MAIN APP FUNCTIONALITY                                                  | #
# -------------------------------------------------------------------------------------------------------------------- #
def app():
    """
    This function uses the sklearn feature extraction function to create a document-term matrix
    """

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                             GLOBAL VARIABLES                                                 | #
    # ---------------------------------------------------------------------------------------------------------------- #
    global DATA, DATA_PATH, ANALYSIS, VERBOSE_DTM, VERBOSITY_DTM, VERBOSE_ANALYSIS, GRANULARITY, SAVE, MODE, \
        STREAMLIT_STATIC_PATH, DOWNLOAD_PATH, DTM, DTM_copy, N, topWords_, DTM_copy_

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                                  INIT                                                        | #
    # ---------------------------------------------------------------------------------------------------------------- #
    if not DOWNLOAD_PATH.is_dir():
        DOWNLOAD_PATH.mkdir()

    st.title('Document-Term Matrix and Word Frequency Analysis')
    st.markdown('## Init\n'
                'Before proceeding, ensure that you have downloaded all the necessary corpora that nltk will be using. '
                'On your terminal, type in the following command into your terminal, which has the same Conda '
                'Virtual Environment as your project activated: ')
    st.code('python -m nltk.downloader all')
    st.markdown('Finally, ensure that the column name of "CLEANED CONTENT" is present in the file you wish to upload. '
                'Failure to do so will result in errors. You may use the template on the left panel to key in and '
                'store your data.')

    st.markdown('## Data Selector')
    SIZE = st.selectbox('Define the size of the file to pass into function', ('Select File Size',
                                                                              'Small File(s)',
                                                                              'Large File(s)'))
    MODE = st.selectbox('Define the data input format', ('CSV', 'XLSX'))

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                              SMALL FILES                                                     | #
    # ---------------------------------------------------------------------------------------------------------------- #
    if SIZE == 'Small File(s)':
        st.markdown('## Load Data\n'
                    'Note that the data passed into this function should be cleaned and most preferably lemmatized or '
                    'stemmed to ensure that the same words in different forms would not be double-counted.')
        DATA_PATH = st.file_uploader('Load up a CSV/XLSX File containing the cleaned data', type=[MODE])

        # ------------------------------------------------------------------------------------------------------------ #
        # |                                                DATA LOADER                                               | #
        # ------------------------------------------------------------------------------------------------------------ #
        if st.button('Load Data', key='data'):
            with st.spinner('Reading Data...'):
                DATA = readFile(DATA_PATH, MODE)
                if not DATA.empty:
                    st.success('Data Loaded!')

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                              LARGE FILES                                                     | #
    # ---------------------------------------------------------------------------------------------------------------- #
    elif SIZE == 'Large File(s)':
        st.markdown('## Load Data\n'
                    'Note that the data passed into this function should be cleaned and most preferably lemmatized or '
                    'stemmed to ensure that the same words in different forms would not be double counted.\n\n'
                    'In the selection boxes below, select the Cloud Service Provider which you have stored the data '
                    'you wish to analyse.')
        CSP = st.selectbox('CSP', ('Choose a CSP', 'Azure', 'Amazon', 'Google'))

        # FUNCTIONALITY FOR FILE RETRIEVAL
        if CSP == 'Azure':
            azure = csp_downloaders.AzureDownloader()
            if st.button('Continue', key='az'):
                azure.downloadBlob()
                DATA = readFile(csp_downloaders.AZURE_DOWNLOAD_ABS_PATH, MODE)

        elif CSP == 'Amazon':
            aws = csp_downloaders.AWSDownloader()
            if st.button('Continue', key='aws'):
                aws.downloadFile()
                DATA = readFile(csp_downloaders.AWS_FILE_NAME, MODE)

        elif CSP == 'Google':
            gcs = csp_downloaders.GoogleDownloader()
            if st.button('Continue', key='gcs'):
                gcs.downloadBlob()
                DATA = readFile(csp_downloaders.GOOGLE_DESTINATION_FILE_NAME, MODE)

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                                    FLAGS                                                     | #
    # ---------------------------------------------------------------------------------------------------------------- #
    st.markdown('## Flags')
    SAVE = st.checkbox('Save Output DataFrame into CSV File?')
    ANALYSIS = st.checkbox('Conduct Analysis on the Document-Term Matrix?')
    if ANALYSIS:
        VERBOSE_DTM = st.checkbox('Display DataFrame of Document-Term Matrix?')
        if VERBOSE_DTM:
            VERBOSITY_DTM = st.slider('Data points to display for Document-Term Matrix?',
                                      min_value=1,
                                      max_value=1000,
                                      value=20)
        VERBOSE_ANALYSIS = st.checkbox('Display top N words in Document-Term Matrix?')
        if VERBOSE_ANALYSIS:
            N = st.slider('Key in the top N number of words to display',
                          key='N',
                          min_value=1,
                          max_value=1000,
                          value=100)
            GRANULARITY = st.number_input('Key in the minimum frequency of words to consider in the data',
                                          key='granularity',
                                          min_value=0,
                                          max_value=1000,
                                          value=2)

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                         DOCUMENT-TERM MATRIX CREATION                                        | #
    # ---------------------------------------------------------------------------------------------------------------- #
    if st.button('Proceed', key='doc'):
        if not DATA.empty:
            with st.spinner('Working to create a Document-Term Matrix...'):
                # INIT COUNTVECTORISER OBJECT -> USED FOR COUNTING THE FREQUENCY OF WORDS THAT APPEAR
                counter_object = CountVectorizer(stop_words=stopwords.words('english'))

                # CREATE STRING CONTAINING THE BAG OF WORDS
                word_list = []
                for index, row in DATA['CLEANED CONTENT'].iteritems():
                    word_list.append(str(row))
                word_string = ' '.join(word_list)

                # CREATE A NEW DF TO PARSE
                dict_data = {
                    'text': word_string
                }
                series_data = pd.DataFrame(data=dict_data, index=[0])

                # FIT-TRANSFORM THE DATA
                series_data = counter_object.fit_transform(series_data.text)

                # CONVERT THE FITTED DATA INTO A PANDAS DATAFRAME TO USE FOR THE DTMs
                DTM = pd.DataFrame(series_data.toarray(),
                                   columns=counter_object.get_feature_names(),
                                   index=[0])

                # ------------------------------------------------------------------------------------------------ #
                # +                                      VISUALISE THE DATA                                      + #
                # ------------------------------------------------------------------------------------------------ #
                if not DTM.empty:
                    if ANALYSIS:
                        if VERBOSE_DTM:
                            # VISUALISE THE DATA
                            st.markdown('## Visualise Data\n'
                                        'The Document-Term Matrix will now be displayed on screen.')
                            DTM_copy = DTM.copy()
                            if VERBOSITY_DTM != 0:
                                try:
                                    st.dataframe(DTM_copy.transpose().head(VERBOSITY_DTM).transpose())
                                except RuntimeError:
                                    st.warning('Warning: Size of DataFrame is too large. Defaulting to 10 data '
                                               'points...')
                                    st.dataframe(DTM_copy.transpose().head(10).transpose())
                            else:
                                try:
                                    st.dataframe(DTM_copy)
                                except RuntimeError:
                                    st.warning('Warning: Size of DataFrame is too large. Defaulting to 10 data '
                                               'points...')
                                    st.dataframe(DTM_copy.transpose().head(10).transpose())

                        if VERBOSE_ANALYSIS:
                            DTM_copy_ = DTM.copy()
                            DTM_copy_ = DTM_copy.transpose()
                            DTM_copy_.sort_values(by=[0], ascending=False, inplace=True)
                            try:
                                st.markdown('The following DataFrame contains the sorted Document-Term Matrix, '
                                            f'displaying the top {N} words of the Document-Term Matrix. Note that '
                                            f'the data displayed here has stopwords removed, hence the usual '
                                            f'words such as "the" is missing.\n')
                                st.dataframe(DTM_copy_.head(N))
                                fig = DTM_copy_.head(N).plot.line(title=f'Frequency of top {N} words used',
                                                                  labels=dict(index='Words',
                                                                              value='Frequency',
                                                                              variable='Count'),
                                                                  width=900,
                                                                  height=500)
                                st.plotly_chart(fig, use_container_width=True)

                                with st.expander("See further elaboration"):
                                    describer = DTM_copy_.describe()
                                    st.write('### Summary of Total Statistics\n\n'
                                             f'**Total number of unique words**: {describer[0]["count"]}\n\n'
                                             f'**Average Frequency of Word Use**: {describer[0]["mean"]}\n\n'
                                             f'**Standard Deviation of Word Use Frequency**: {describer[0]["std"]}\n\n'
                                             )
                            except RuntimeError:
                                st.warning('Warning: DataFrame is too large to display. '
                                           'Defaulting to 10 data points...')
                                st.dataframe(DTM_copy_.head(10))

                    if SAVE:
                        st.markdown('## Download Data')
                        if VERBOSE_DTM:
                            st.markdown('### Document-Term Matrix')
                            st.markdown('Download requested data from [downloads/DTM.csv](downloads/DTM.csv)')
                            DTM.to_csv(str(DOWNLOAD_PATH / 'DTM.csv'), index=False)
                        if VERBOSE_ANALYSIS:
                            st.markdown('### Document-Term Matrix: Sorted')
                            st.markdown('Download requested data from [downloads/sorted_DTM.csv](downloads/'
                                        'sorted_DTM.csv)')
                            DTM_copy_.to_csv(str(DOWNLOAD_PATH / 'sorted_DTM.csv'))
        else:
            st.error('Error: No files are loaded.')
