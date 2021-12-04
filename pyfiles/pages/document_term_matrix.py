"""
Document Term Matrix is one of the core modules of this app.

This module is responsible for the creation of a Document-Term Matrix and visualising the Document-Term Matrix.

A large portion of this module uses the nltk and scikit-learn packages to create the Document-Term Matrix document.
"""

# -------------------------------------------------------------------------------------------------------------------- #
# |                                             IMPORT RELEVANT LIBRARIES                                            | #
# -------------------------------------------------------------------------------------------------------------------- #
import os
import pathlib

import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from utils import csp_downloaders
from utils.helper import readFile, printDataFrame

# -------------------------------------------------------------------------------------------------------------------- #
# |                                           GLOBAL VARIABLE DECLARATION                                            | #
# -------------------------------------------------------------------------------------------------------------------- #
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
DOWNLOAD_PATH = (STREAMLIT_STATIC_PATH / "downloads")
DTM = pd.DataFrame()
pd.options.plotting.backend = 'plotly'
FILE = 'Small File(s)'
DATA = pd.DataFrame()
DATA_PATH = None
ANALYSIS = False
VERBOSE_DTM = False
VERBOSITY_DTM = False
VERBOSE_ANALYSIS = False
SAVE = False
MODE = 'CSV'
DTM_copy = pd.DataFrame()
N = 100
CSP = None
ADVANCED_ANALYSIS = False
FINALISED_DATA_LIST = []
DATA_COLUMN = None
TOP_N_WORD_FIG = None
FC = 0


# -------------------------------------------------------------------------------------------------------------------- #
# |                                             MAIN APP FUNCTIONALITY                                               | #
# -------------------------------------------------------------------------------------------------------------------- #
def app():
    """
    Main function that will be called when the app is run and this module is called
    """

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                GLOBAL VARIABLES                                                  | #
# -------------------------------------------------------------------------------------------------------------------- #
    global DATA, DATA_PATH, ANALYSIS, VERBOSE_DTM, VERBOSITY_DTM, VERBOSE_ANALYSIS, SAVE, FILE, MODE, \
        STREAMLIT_STATIC_PATH, DOWNLOAD_PATH, DTM, DTM_copy, N, DTM_copy, CSP, ADVANCED_ANALYSIS, \
        FINALISED_DATA_LIST, DATA_COLUMN, TOP_N_WORD_FIG, FC

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                     INIT                                                         | #
# -------------------------------------------------------------------------------------------------------------------- #
    if not DOWNLOAD_PATH.is_dir():
        DOWNLOAD_PATH.mkdir()

    st.title('Document-Term Matrix and Word Frequency Analysis')
    st.markdown('## Init\n'
                'Before proceeding, you will need to download the corpus needed to process your data. To do so, click '
                'on the "Begin Download" button below. Please ensure that you have at least 3 GB of free disk space '
                'available so that you are able to download the corpus onto your system and that your device is '
                'connected to the Internet.')
    if st.button('Begin Download', key='download-model'):
        os.system('python -m nltk.downloader all')
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
                'Ensure that your data is cleaned and lemmatized before passing it into this module. If '
                'you have not cleaned your dataset, use the *Load, Clean and Visualise* module to clean up your '
                'data before proceeding. Do not upload the a file containing tokenized data for DTM creation.')
    FILE = st.selectbox('Select the Size of File to Load', ('Small File(s)', 'Large File(s)'))
    MODE = st.selectbox('Define the Data Input Format', ('CSV', ' XLSX'))

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                 FILE UPLOADING                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    if FILE == 'Small File(s)':
        st.markdown('### Upload File\n')
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
# |                                                      FLAGS                                                       | #
# -------------------------------------------------------------------------------------------------------------------- #
    st.markdown('## Flags\n'
                'The following section contains flags you can use to better define the outputs of this module and to '
                'modify what the app prints out to screen.\n\n'
                'Note that there is an size limit **(50 MB)** for the DataFrames that are printed to screen. If '
                'you get an error telling you that the DataFrame size is too large to proceed, kindly lower the number '
                'of data points you wish to visualise or download the file and visualise it through Excel or any other '
                'DataFrame visualising Python packages. There is no definitive way to increase the size of the '
                'DataFrame that can be printed out due to the inherent limitation on the size of the packets sent '
                'over to and from the Streamlit server.')
    SAVE = st.checkbox('Save Outputs?', help='Due to the possibility of files with the same file name and '
                                             'content being downloaded again, a unique file identifier is '
                                             'tacked onto the filename.')
    VERBOSE_DTM = st.checkbox('Display DataFrame of Document-Term Matrix?')
    if VERBOSE_DTM:
        VERBOSITY_DTM = st.slider('Data Points to Display for Document-Term Matrix?',
                                  min_value=0,
                                  max_value=1000,
                                  value=100,
                                  help='Note that this parameter defines the number of points to print out for '
                                       'the raw DTM produced by the app; Select 0 to display all Data Points')
        VERBOSE_ANALYSIS = st.checkbox('Display top N words in Document-Term Matrix?',
                                       help='This sorts the DTM and shows you the top number of words you select.')
        if VERBOSE_ANALYSIS:
            N = st.slider('Key in the top N number of words to display',
                          key='N',
                          min_value=0,
                          max_value=1000,
                          value=100,
                          help='This parameter controls the top number of words that will be displayed and plotted. '
                               'This parameter is not the same as that above which controls the number of data points '
                               'printed out for the raw DTM DataFrame; Select 0 to display all Data Points')
        ADVANCED_ANALYSIS = st.checkbox('Display Advanced DataFrame Statistics?',
                                        help='This option will analyse your DataFrame and display advanced '
                                             'statistics on it. Note that this will require some time and '
                                             'processing power to complete. Deselect this option if this if '
                                             'you do not require it.')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                            DOCUMENT-TERM MATRIX CREATION                                         | #
# -------------------------------------------------------------------------------------------------------------------- #
    st.markdown('---')
    st.markdown('## Document-Term Matrix Creation\n'
                'Ensure that you have successfully uploaded the data and selected the correct field for analysis '
                'before clicking on the "Proceed" button. The status of your file upload is displayed below for your '
                'reference.')
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

    if st.button('Proceed', key='doc'):
        if (FILE == 'Small File(s)' and DATA_PATH) or (FILE == 'Large File(s)' and not DATA.empty):
            st.info('Data loaded properly!')
            DATA = DATA.astype(str)
            with st.spinner('Working to create a Document-Term Matrix...'):
                counter_object = CountVectorizer(stop_words=stopwords.words('english'))

                # CONCATENATE ALL STR VALUES INTO LIST: OPTIMISED
                word_string = ' '.join(DATA[DATA_COLUMN])

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

# -------------------------------------------------------------------------------------------------------------------- #
# +                                                 VISUALISE THE DATA                                               + #
# -------------------------------------------------------------------------------------------------------------------- #
            if not DTM.empty:
                DTM_copy = DTM.copy().transpose()
                DTM_copy.columns = ['Word Frequency']
                DTM_copy.sort_values(by=['Word Frequency'], ascending=False, inplace=True)
                TOP_N_WORD_FIG = DTM_copy.head(N).plot.line(title=f'Frequency of top {N} words used',
                                                            labels=dict(index='Word',
                                                                        value='Frequency',
                                                                        variable='Count'),
                                                            width=900,
                                                            height=500)

                if VERBOSE_DTM:
                    # VISUALISE THE DATA
                    st.markdown('## DTM Data\n'
                                'The Document-Term Matrix will now be displayed on screen.')
                    try:
                        st.dataframe(DTM.transpose().head(VERBOSITY_DTM).transpose(), width=900, height=500)
                    except RuntimeError:
                        st.warning('Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                        st.dataframe(DTM.transpose().head(10).transpose(), height=600, width=800)
                    except Exception as ex:
                        st.error(f'Error: {ex}')

                    if VERBOSE_ANALYSIS:
                        st.markdown(f'## Top {N} Words in DTM\n'
                                    'The following DataFrame contains the sorted Document-Term Matrix, '
                                    f'displaying the top {N} words of the Document-Term Matrix.')
                        printDataFrame(data=DTM_copy, verbose_level=N, advanced=ADVANCED_ANALYSIS)
                        st.plotly_chart(TOP_N_WORD_FIG, use_container_width=True)
                        with st.expander('Elaboration'):
                            st.markdown('If your data is sufficiently large enough, you may see that your data '
                                        'resembles an exponential graph. This phenomenon is known as **Zipf\'s Law**. '
                                        '\n\nZipf\'s Law states that given a corpus of natural language utterances, '
                                        'the frequency of any word in the corpus is inversely proportional to the '
                                        'word\'s rank in the frequency table, i.e. **f ∝ 1 / rank**.\n\n'
                                        'Naturally, to find the frequency of word use in any corpus, we simply '
                                        'multiply the inverse of the rank with frequency of the most used word. '
                                        'This value provides a good approximation of the actual frequency of word use '
                                        'in the corpus.')

                # FINALISED DATA LIST
                if not VERBOSE_DTM and not VERBOSE_ANALYSIS:
                    FINALISED_DATA_LIST = [
                        (DTM, 'DTM', 'dtm.csv', 'csv')
                    ]
                elif VERBOSE_DTM and not VERBOSE_ANALYSIS:
                    FINALISED_DATA_LIST = [
                        (DTM, 'DTM', 'dtm.csv', 'csv'),
                        (DTM_copy, 'Sorted DTM', 'sorted_dtm.csv', 'csv_')
                    ]
                if VERBOSE_DTM and VERBOSE_ANALYSIS:
                    FINALISED_DATA_LIST = [
                        (DTM, 'DTM', 'dtm.csv', 'csv'),
                        (DTM_copy, 'Sorted DTM', 'sorted_dtm.csv', 'csv_'),
                        (TOP_N_WORD_FIG, f'Top {N} Words Frequency', 'top_n.png', 'png')
                    ]

# -------------------------------------------------------------------------------------------------------------------- #
# +                                                   SAVE THE DATA                                                  + #
# -------------------------------------------------------------------------------------------------------------------- #
                if SAVE:
                    st.markdown('---')
                    st.markdown('## Download Data')
                    for data in FINALISED_DATA_LIST:
                        if data[3] == 'csv':
                            if not data[0].empty:
                                st.markdown(f'### {data[1]}')
                                st.markdown(f'Download requested data from [downloads/{data[2]}]'
                                            f'(downloads/id{FC}_{data[2]})')
                                data[0].to_csv(str(DOWNLOAD_PATH / f'id{FC}_{data[2]}'), index=False)
                                FC += 1
                        elif data[3] == 'csv_':
                            if not data[0].empty:
                                st.markdown(f'### {data[1]}')
                                st.markdown(f'Download requested data from [downloads/{data[2]}]'
                                            f'(downloads/id{FC}_{data[2]})')
                                data[0].to_csv(str(DOWNLOAD_PATH / f'id{FC}_{data[2]}'), index=True)
                                FC += 1
                        elif data[3] == 'png':
                            if data[0]:
                                st.markdown(f'### {data[1]}')
                                st.markdown(f'Download requested data from [downloads/{data[2]}]'
                                            f'(downloads/id{FC}_{data[2]})')
                                data[0].write_image(str(DOWNLOAD_PATH / f'id{FC}_{data[2]}'))
                                FC += 1
            else:
                st.error('Error: DTM not created properly. Try again.')
        else:
            st.error('Error: No files are loaded.')
