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
        STREAMLIT_STATIC_PATH, DOWNLOAD_PATH, DTM, DTM_copy, N, topWords_, DTM_copy_, CSP

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                                  INIT                                                        | #
    # ---------------------------------------------------------------------------------------------------------------- #
    if not DOWNLOAD_PATH.is_dir():
        DOWNLOAD_PATH.mkdir()

    st.title('Document-Term Matrix and Word Frequency Analysis')
    st.markdown('## Init\n'
                'Before proceeding, you will need to download the corpus needed to process your data. To do so, run '
                'the following command on your terminal. Please ensure that you have at least 3 GB of free disk space '
                'available so that you are able to download the corpus onto your system.')
    st.code('python -m nltk.downloader all')
    st.markdown('For this module, you need to use the *DTM* Template displayed on the left. If you do not wish to '
                'use the template for any reason, ensure that the column name "CLEANED CONTENT" is in the data you are '
                'passing into the app, and that the column contains data that you wish to clean or visualise.')

    st.markdown('## Data Format')
    SIZE = st.selectbox('Define the size of the file to pass into function', ('Select File Size',
                                                                              'Small File(s)',
                                                                              'Large File(s)'))
    MODE = st.selectbox('Define the data input format', ('CSV', 'XLSX'))

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                              SMALL FILES                                                     | #
    # ---------------------------------------------------------------------------------------------------------------- #
    if SIZE == 'Small File(s)':
        st.markdown('## Load Data\n'
                    'For this mode, ensure that your file is smaller than 200 MB in size. If your file is larger than '
                    '200 MB, you may choose to rerun the app with the tag `--server.maxUploadSize=[SIZE_IN_MB_HERE]` '
                    'appended behind the `streamlit run app.py` command and define the maximum size of file you can '
                    'upload onto Streamlit, or use the Large File option to pull your dataset from any one of the '
                    'three supported Cloud Service Providers into the app. Note that modifying the command you use '
                    'to run the app is not available if you are using the web interface for the app and you will be '
                    'limited to using the Large File option to pull datasets larger than 200 MB in size.\n\n'
                    'Also, ensure that your data is cleaned and lemmatized before passing it into this module. If '
                    'you have not cleaned your dataset, use the Load, Clean and Visualise module to clean up your '
                    'data before proceeding.')
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
                    'stemmed before proceeding.\n\n'
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
    st.markdown('## Flags\n'
                'The following section contains flags you can use to better define the outputs of this module and to '
                'modify what the app prints out to screen.\n\n'
                'Note that there is an inherent size limit (50 MB) for the DataFrames that are printed to screen. If '
                'you get an error telling you that the DataFrame size is too large to proceed, kindly lower the number '
                'of data points you wish to visualise or download the file and visualise it through Excel or any other '
                'DataFrame visualising Python packages.')
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
            st.markdown('Note that the following options will allow you to modify the minimum frequency of words to '
                        'consider in your dataset. This feature is added to allow you to remove outliers in your data '
                        '(e.g. words that appear once in the entire document and never again, spelling errors that '
                        'does not appear anywhere else in the text, etc.) that you may not want to '
                        'analyse. If you do not wish to incorporate this behaviour into your data, leave the minimum '
                        'frequency setting at 1 and modify only the top N words to display.')
            GRANULARITY = st.number_input('Key in the minimum frequency of words to consider [WARNING: MODIFIES '
                                          'OUTPUT]',
                                          key='granularity',
                                          min_value=1,
                                          max_value=1000,
                                          value=1)
            N = st.slider('Key in the top N number of words to display',
                          key='N',
                          min_value=1,
                          max_value=1000,
                          value=100)

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
                                            f'displaying the top {N} words of the Document-Term Matrix.')
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
