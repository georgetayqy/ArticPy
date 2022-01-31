"""
Document Term Matrix is one of the core modules of this app.
This module is responsible for the creation of a Document-Term Matrix and visualising the Document-Term Matrix.
A large portion of this module uses the nltk and scikit-learn packages to create the Document-Term Matrix document.
"""

# -------------------------------------------------------------------------------------------------------------------- #
# |                                             IMPORT RELEVANT LIBRARIES                                            | #
# -------------------------------------------------------------------------------------------------------------------- #
import io
import os
import pathlib
import platform
import pandas as pd
import plotly.graph_objects
import streamlit as st

from config import dtm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from utils import csp_downloaders
from utils.helper import readFile, printDataFrame, prettyDownload
pd.options.plotting.backend = 'plotly'


# -------------------------------------------------------------------------------------------------------------------- #
# |                                             MAIN APP FUNCTIONALITY                                               | #
# -------------------------------------------------------------------------------------------------------------------- #
def app():
    """
    Main function that will be called when the app is run and this module is called
    """

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                     INIT                                                         | #
# -------------------------------------------------------------------------------------------------------------------- #
    st.title('Document-Term Matrix and Word Frequency Analysis')
    with st.expander('Module Description'):
        st.markdown('Before proceeding, you will need to download the corpus needed to process your data. To do so, '
                    'click on the "Begin Download" button below. Please ensure that you have at least 3 GB of free '
                    'disk space available so that you are able to download the corpus onto your system and that your '
                    'device is connected to the Internet.')

        # CHECK IF THE DATA HAS BEEN DOWNLOADED
        if platform.system() == 'Windows':
            if pathlib.Path.joinpath(pathlib.Path.home(), 'AppData', 'Roaming', 'nltk_data').is_dir():
                if any(pathlib.Path.joinpath(pathlib.Path.home(), 'AppData', 'Roaming', 'nltk_data').iterdir()):
                    st.info('NTLK Data Detected')
                else:
                    st.warning('NLTK Data Not Detected')
            else:
                st.warning('NLTK Data Not Detected')
        elif platform.system() == 'Linux':
            if pathlib.Path.joinpath(pathlib.Path.home(), 'local', 'share', 'nltk_data').is_dir():
                if any(pathlib.Path.joinpath(pathlib.Path.home(), 'local', 'share', 'nltk_data').iterdir()):
                    st.info('NTLK Data Detected')
                else:
                    st.warning('NLTK Data Not Detected')
            else:
                st.warning('NLTK Data Not Detected')
        elif platform.system() == 'Darwin':
            if pathlib.Path.joinpath(pathlib.Path.home(), 'nltk_data').is_dir():
                if any(pathlib.Path.joinpath(pathlib.Path.home(), 'nltk_data').iterdir()):
                    st.info('NTLK Data Detected')
                else:
                    st.warning('NLTK Data Not Detected')
            else:
                st.warning('NLTK Data Not Detected')

        if st.button('Begin Download', key='download-model'):
            os.system('python -m nltk.downloader all')

    st.markdown('## Upload Data\n')
    col1, col1_ = st.columns(2)
    dtm['FILE'] = col1.selectbox('Origin of Data File', ('Local', 'Online'),
                                 help='Choose "Local" if you wish to upload a file from your machine or choose '
                                      '"Online" if you wish to pull a file from any one of the supported Cloud '
                                      'Service Providers.',
                                 key='dtm-file')
    dtm['MODE'] = col1_.selectbox('Define the Data Input Format', ('CSV', 'XLSX', 'PKL', 'JSON'), key='dtm-mode')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                 FILE UPLOADING                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    if dtm['FILE'] == 'Local':
        dtm['DATA_PATH'] = st.file_uploader(f'Load {dtm["MODE"]} File', type=[dtm['MODE']], key='dtm-fp')
        if dtm['DATA_PATH'] is not None:
            dtm['DATA'] = readFile(dtm['DATA_PATH'], dtm['MODE'])
            if not dtm['DATA'].empty or not dtm['DATA']:
                dtm['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored', list(dtm['DATA'].columns),
                                                  key='dtm-dc')
                st.success(f'Data Loaded from {dtm["DATA_COLUMN"]}!')
        else:
            # RESET
            st.warning('Warning: Your Dataset file is not loaded.')
            dtm['DATA'] = pd.DataFrame()

    elif dtm['FILE'] == 'Online':
        st.info(f'File Format Selected: **{dtm["MODE"]}**')
        dtm['CSP'] = st.selectbox('CSP', ('Select a CSP', 'Azure', 'Amazon', 'Google'), key='dtm-csp')

        if dtm['CSP'] == 'Azure':
            azure = csp_downloaders.AzureDownloader()
            if azure.SUCCESSFUL:
                try:
                    azure.downloadBlob()
                    dtm['DATA'] = readFile(azure.AZURE_DOWNLOAD_PATH, dtm['MODE'])
                except Exception as ex:
                    dtm['DATA'] = pd.DataFrame()
                    st.error(f'Error: {ex}. Try again.')

            if not dtm['DATA'].empty:
                dtm['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored', list(dtm['DATA'].columns),
                                                  key='dtm-az')
                st.success(f'Data Loaded from {dtm["DATA_COLUMN"]}!')

        elif dtm['CSP'] == 'Amazon':
            aws = csp_downloaders.AWSDownloader()
            if aws.SUCCESSFUL:
                try:
                    aws.downloadFile()
                    dtm['DATA'] = readFile(aws.AWS_FILE_NAME, dtm['MODE'])
                except Exception as ex:
                    dtm['DATA'] = pd.DataFrame()
                    st.error(f'Error: {ex}. Try again.')

            if not dtm['DATA'].empty:
                dtm["DATA_COLUMN"] = st.selectbox('Choose Column where Data is Stored', list(dtm['DATA'].columns),
                                                  key='dtm-aws')
                st.success(f'Data Loaded from {dtm["DATA_COLUMN"]}!')

        elif dtm['CSP'] == 'Google':
            gcs = csp_downloaders.GoogleDownloader()
            if gcs.SUCCESSFUL:
                try:
                    gcs.downloadBlob()
                    dtm['DATA'] = readFile(gcs.GOOGLE_DESTINATION_FILE_NAME, dtm['MODE'])
                except Exception as ex:
                    dtm['DATA'] = pd.DataFrame()
                    st.error(f'Error: {ex}. Try again.')

            if not dtm['DATA'].empty:
                dtm["DATA_COLUMN"] = st.selectbox('Choose Column where Data is Stored', list(dtm['DATA'].columns),
                                                  key='dtm-gcs')
                st.success(f'Data Loaded from {dtm["DATA_COLUMN"]}!')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                      FLAGS                                                       | #
# -------------------------------------------------------------------------------------------------------------------- #
    st.markdown('## Options\n'
                'The following section contains options you can use to better define the outputs of this module and to '
                'modify what the app prints out to screen.\n\n')
    dtm['SAVE'] = st.checkbox('Save Outputs?', help='The files outputted will be in the same format as the input '
                                                    'file format by default but this behaviour can be overridden.',
                              key='dtm-save')
    if dtm['SAVE']:
        if st.checkbox('Override Output Format?', key='override'):
            dtm['OVERRIDE_FORMAT'] = st.selectbox('Overridden Output Format', ('CSV', 'XLSX', 'PKL', 'JSON'),
                                                  key='override-format')
            if dtm['OVERRIDE_FORMAT'] == dtm['MODE']:
                st.warning('Warning: Overridden Format is the same as Input Format')
        else:
            dtm['OVERRIDE_FORMAT'] = None

    dtm['VERBOSE_DTM'] = st.checkbox('Display DataFrame of Document-Term Matrix?',
                                     help='Note that there is an size limit for the DataFrames that are '
                                          'printed to screen. If you get an error telling you that the DataFrame size '
                                          'is too large to proceed, kindly lower the number of data points you wish '
                                          'to visualise or increase the maximum size of items to print to screen '
                                          'through the maxMessageSize setting in the Streamlit config file.',
                                     key='vb')
    if dtm['VERBOSE_DTM']:
        dtm['VERBOSITY_DTM'] = st.slider('Data Points to Display for Document-Term Matrix?',
                                         min_value=1,
                                         max_value=1000,
                                         value=100,
                                         help='Note that this parameter defines the number of points to print out '
                                              'for the raw DTM produced by the app. You can only display up to a '
                                              'maximum of 1000 data points and a minimum of 1 data point at any one '
                                              'time.',
                                         key='vbsity')
        dtm['VERBOSE_ANALYSIS'] = st.checkbox('Display top N words in Document-Term Matrix?',
                                              help='This sorts the DTM and shows you the top number of words you '
                                                   'select.',
                                              key='vbanalysis')
        if dtm['VERBOSE_ANALYSIS']:
            dtm['N'] = st.slider('Key in the top N number of words to display',
                                 key='N',
                                 min_value=0,
                                 max_value=1000,
                                 value=100,
                                 help='This parameter controls the top number of words that will be displayed and '
                                      'plotted. This parameter is not the same as that above which controls the number '
                                      'of data points printed out for the raw DTM DataFrame; Select 0 to display all '
                                      'Data Points')
        dtm['ADVANCED_ANALYSIS'] = st.checkbox('Display Advanced DataFrame Statistics?',
                                               help='This option will analyse your DataFrame and display advanced '
                                               'statistics on it. Note that this will require some time and '
                                               'processing power to complete. Deselect this option if this if '
                                               'you do not require it.',
                                               key='dtm-advanced-df')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                            DOCUMENT-TERM MATRIX CREATION                                         | #
# -------------------------------------------------------------------------------------------------------------------- #
    st.markdown('---')
    st.markdown('## Document-Term Matrix Creation\n'
                'Ensure that you have successfully uploaded the data and selected the correct field for analysis '
                'before clicking on the "Proceed" button.')

    if dtm['FILE'] == 'Local':
        if dtm['DATA_PATH']:
            st.info('File loaded.')
        else:
            st.warning('File has not been loaded.')
    elif dtm['FILE'] == 'Online':
        if not dtm['DATA'].empty:
            st.info('File loaded.')
        else:
            st.warning('File has not been loaded.')

    if st.button('Proceed', key='dtm'):
        if (dtm['FILE'] == 'Local' and dtm['DATA_PATH']) or \
                (dtm['FILE'] == 'Online' and not dtm['DATA'].empty):
            st.info('Data loaded properly!')
            dtm['DATA'] = dtm['DATA'].astype(str)
            with st.spinner('Working to create a Document-Term Matrix...'):
                counter_object = CountVectorizer(stop_words=stopwords.words('english'))

                # CONCATENATE ALL STR VALUES INTO LIST: OPTIMISED
                word_string = ' '.join(dtm['DATA'][dtm["DATA_COLUMN"]])

                # CREATE A NEW DF TO PARSE
                dict_data = {
                    'text': word_string
                }
                series_data = pd.DataFrame(data=dict_data, index=[0])

                # FIT-TRANSFORM THE DATA
                series_data = counter_object.fit_transform(series_data.text)

                # CONVERT THE FITTED DATA INTO A PANDAS DATAFRAME TO USE FOR THE DTMs
                dtm['DTM'] = pd.DataFrame(series_data.toarray(),
                                          columns=counter_object.get_feature_names(),
                                          index=[0])

# -------------------------------------------------------------------------------------------------------------------- #
# +                                                 VISUALISE THE DATA                                               + #
# -------------------------------------------------------------------------------------------------------------------- #
            if not dtm['DTM'].empty:
                dtm['DTM_copy'] = dtm['DTM'].copy().transpose()
                dtm['DTM_copy'].columns = ['Word Frequency']
                dtm['DTM_copy'].sort_values(by=['Word Frequency'], ascending=False, inplace=True)
                dtm['TOP_N_WORD_FIG'] = dtm['DTM_copy'].head(dtm["N"]).plot.line(title=f'Frequency of top {dtm["N"]} '
                                                                                 f'words used',
                                                                                 labels=dict(index='Word',
                                                                                             value='Frequency',
                                                                                             variable='Count'),
                                                                                 width=900,
                                                                                 height=500)

                if dtm['VERBOSE_DTM']:
                    # VISUALISE THE DATA
                    st.markdown('### DTM Data\n'
                                'The Document-Term Matrix will now be displayed on screen.')
                    try:
                        st.dataframe(dtm['DTM'].transpose().head(dtm["VERBOSITY_DTM"]).transpose(),
                                     width=900,
                                     height=500)
                    except RuntimeError:
                        st.warning('Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                        st.dataframe(dtm['DTM'].transpose().head(10).transpose(), height=600, width=800)
                    except Exception as ex:
                        st.error(f'Error: {ex}')

                    if dtm['VERBOSE_ANALYSIS']:
                        st.markdown(f'### Top {dtm["N"]} Words in DTM\n'
                                    'The following DataFrame contains the sorted Document-Term Matrix, displaying the '
                                    f'top {dtm["N"]} words of the Document-Term Matrix. If 0 is seen above, you have '
                                    'chose to print out the entire DataFrame onto the screen.')
                        printDataFrame(data=dtm['DTM_copy'], verbose_level=dtm["N"], advanced=dtm['ADVANCED_ANALYSIS'])
                        st.plotly_chart(dtm['TOP_N_WORD_FIG'], use_container_width=True)
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
                if not dtm['VERBOSE_DTM'] and not dtm['VERBOSE_ANALYSIS']:
                    dtm['FINALISED_DATA_LIST'] = [
                        (dtm['DTM'], 'DTM', 'dtm', '.csv', False)
                    ]
                elif dtm['VERBOSE_DTM'] and not dtm['VERBOSE_ANALYSIS']:
                    dtm['FINALISED_DATA_LIST'] = [
                        (dtm['DTM'], 'DTM', 'dtm', '.csv', False),
                        (dtm['DTM_copy'], 'Sorted DTM', 'sorted_dtm', '.csv', True)
                    ]
                if dtm['VERBOSE_DTM'] and dtm['VERBOSE_ANALYSIS']:
                    dtm['FINALISED_DATA_LIST'] = [
                        (dtm['DTM'], 'DTM', 'dtm', '.csv', False),
                        (dtm['DTM_copy'], 'Sorted DTM', 'sorted_dtm', '.csv', True),
                        (dtm['TOP_N_WORD_FIG'], f'Top {dtm["N"]} Words Frequency', 'top_n', '.png', False)
                    ]

# -------------------------------------------------------------------------------------------------------------------- #
# +                                                   SAVE THE DATA                                                  + #
# -------------------------------------------------------------------------------------------------------------------- #
                if dtm['SAVE']:
                    st.markdown('---')
                    st.markdown('### Download Data')
                    for index, data in enumerate(dtm['FINALISED_DATA_LIST']):
                        if dtm['OVERRIDE_FORMAT'] is not None:
                            if data[3] != '.png':
                                st.markdown(prettyDownload(object_to_download=data[0],
                                                           download_filename=f'{data[2]}.'
                                                                             f'{dtm["OVERRIDE_FORMAT"].lower()}',
                                                           button_text=f'Download {data[1]} Data',
                                                           override_index=data[4],
                                                           format_=dtm['OVERRIDE_FORMAT']),
                                            unsafe_allow_html=True)
                            else:
                                st.markdown(prettyDownload(object_to_download=data[0],
                                                           download_filename=f'{data[2]}.png',
                                                           button_text=f'Download {data[1]} Data',
                                                           override_index=data[4]),
                                            unsafe_allow_html=True)
                        else:
                            if data[3] != '.png':
                                st.markdown(prettyDownload(object_to_download=data[0],
                                                           download_filename=f'{data[2]}{data[3]}',
                                                           button_text=f'Download {data[1]} Data',
                                                           override_index=data[4],
                                                           format_=dtm["MODE"]),
                                            unsafe_allow_html=True)
                            else:
                                st.markdown(prettyDownload(object_to_download=data[0],
                                                           download_filename=f'{data[2]}.png',
                                                           button_text=f'Download {data[1]} Data',
                                                           override_index=data[4]),
                                            unsafe_allow_html=True)
            else:
                st.error('Error: DTM not created properly. Try again.')
        else:
            st.error('Error: No files are loaded.')
