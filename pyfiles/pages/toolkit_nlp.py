"""
This module allows the user to conduct basic NLP analysis functions on the preprocessed data using the spaCy module

This module uses CPU-optimised pipelines and hence a GPU is optional in this module
"""

# -------------------------------------------------------------------------------------------------------------------- #
# |                                         IMPORT RELEVANT LIBRARIES                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
import multiprocessing
import os
import pathlib
import numpy as np
import pandas as pd
import spacy
import streamlit as st
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
import nltk
import kaleido
import pyLDAvis
import pyLDAvis.gensim_models
import pandas_profiling
import pyLDAvis.sklearn
import streamlit.components.v1

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from streamlit_pandas_profiling import st_profile_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from spacy import displacy
from wordcloud import WordCloud
from textblob import TextBlob
from utils import csp_downloaders
from utils.helper import readFile, summarise, modelIterator, printDataFrame, dominantTopic, modelNMFIterator

# -------------------------------------------------------------------------------------------------------------------- #
# |                                         GLOBAL VARIABLE DECLARATION                                              | #
# -------------------------------------------------------------------------------------------------------------------- #
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
DOWNLOAD_PATH = (STREAMLIT_STATIC_PATH / "downloads")
DATA = pd.DataFrame()
FILE = 'Small File(s)'
MODE = 'CSV'
DATA_PATH = None
CSP = None
SAVE = False
VERBOSE = False
VERBOSITY = 20
APP_MODE = 'Wordcloud'
BACKEND_ANALYSER = 'VADER'
MAX_WORDS = 200
CONTOUR_WIDTH = 3
HEIGHT = 400
WIDTH = 800
SENT_LEN = 3
NUM_TOPICS = 10
TOPIC_FRAME = None
LDA_DATA = pd.DataFrame()
LDA_VIS = None
LDA_MODEL = None
KW = None
TFIDF_MODEL = None
TFIDF_VECTORISED = None
NMF_MODEL = None
NMF_DATA = None
LSI_MODEL = None
LSI_DATA = None
MAR_FIG = None
WORD_FIG = None
LDA_VIS_STR = None
MODEL = None
ADVANCED_ANALYSIS = False
NLP_MODEL = 'en_core_web_sm'
DATA_COLUMN = None
NLP = None
ONE_DATAPOINT = False
DATAPOINT_SELECTOR = 0
NLP_TOPIC_MODEL = 'Latent Dirichlet Allocation'
MIN_DF = 5
MAX_DF = 0.90
MAX_ITER = 10
CV = None
VECTORISED = None
COLOUR = None
TOPIC_TEXT = []
SVG = None
HAC_PLOT = None
WORKER = 1
MAX_FEATURES = 5000
ALPHA = 0.1
L1_RATIO = 0.5


# -------------------------------------------------------------------------------------------------------------------- #
# |                                             MAIN APP FUNCTIONALITY                                               | #
# -------------------------------------------------------------------------------------------------------------------- #
def app():
    """
    Main function that will be called when the app is run
    """

# -------------------------------------------------------------------------------------------------------------------- #
# |                                               GLOBAL VARIABLES                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    global FILE, MODE, DATA_PATH, CSP, SAVE, VERBOSE, VERBOSITY, APP_MODE, BACKEND_ANALYSER, MAX_WORDS, \
        CONTOUR_WIDTH, DATA, SENT_LEN, NUM_TOPICS, LDA_MODEL, MODEL, LDA_DATA, LDA_VIS, ADVANCED_ANALYSIS, \
        NLP_MODEL, DATA_COLUMN, NLP, ONE_DATAPOINT, DATAPOINT_SELECTOR, NLP_TOPIC_MODEL, MIN_DF, MAX_DF, MAX_ITER, \
        NMF_MODEL, NMF_DATA, LSI_MODEL, LSI_DATA, TFIDF_MODEL, TFIDF_VECTORISED, MAR_FIG, WORD_FIG, CV, VECTORISED, \
        COLOUR, TOPIC_TEXT, LDA_VIS_STR, WIDTH, HEIGHT, SVG, HAC_PLOT, WORKER, MAX_FEATURES, KW, TOPIC_FRAME, ALPHA, \
        L1_RATIO

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                    INIT                                                          | #
# -------------------------------------------------------------------------------------------------------------------- #
    # CREATE THE DOWNLOAD PATH WHERE FILES CREATED WILL BE STORED AND AVAILABLE FOR DOWNLOADING
    if not DOWNLOAD_PATH.is_dir():
        DOWNLOAD_PATH.mkdir()

    st.title('NLP Toolkit')
    st.markdown('## Init\n'
                'This module uses the spaCy package to conduct the necessary NLP preprocessing and '
                'analysis tasks for users to make sense of the text they pass into app. Note that this app requires '
                'the data to be decently cleaned; if you have not done so, run the Load, Clean adn Visualise module '
                'and save the cleaned  data onto your workstation. Those files may come in useful in '
                'the functionality of this app.\n\n')
    st.markdown('## Upload Data\n'
                'Due to limitations imposed by the file uploader widget, only files smaller than 200 MB can be loaded '
                'with the widget. If your files are larger than 200 MB, please select "Large File(s)" and select the '
                'CSP you are using to host and store your data. To circumvent this limitation, you may choose to '
                'rerun the app with the tag `--server.maxUploadSize=[SIZE_IN_MB_HERE]` appended behind the '
                '`streamlit run app.py` command and define the maximum size of file you can upload '
                'onto Streamlit, or use the Large File option to pull your dataset from any one of the three supported '
                'Cloud Service Providers into the app. Note that modifying the command you use to run the app is not '
                'available if you are using the **web interface** for the app and you will be limited to using the '
                'Large File option to pull datasets larger than 200 MB in size. For Docker, you will need to append '
                'the tag above behind the Docker Image name when running the *run* command, e.g. '
                '`docker run asdfghjklxl/news:latest --server.maxUploadSize=1028`.\n\n'
                'After uploading your files, select the file format you wish to upload. You are warned that if you '
                'fail to define the correct file format you wish to upload, the app will not let you upload it '
                '(if you are using the Small File module and may result in errors (for Large File module).\n\n')
    FILE = st.selectbox('Select the type of file to load', ('Small File(s)', 'Large File(s)'))
    MODE = st.selectbox('Define the data input format', ('CSV', 'XLSX'))

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                 FILE UPLOADING                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    st.markdown('### Upload the file that you wish to analyse:\n')
    if FILE == 'Small File(s)':
        DATA_PATH = st.file_uploader(f'Load up a {MODE} File containing the cleaned data', type=[MODE])
        if DATA_PATH is not None:
            DATA = readFile(DATA_PATH, MODE)
            if not DATA.empty:
                DATA = DATA.astype(str)
                DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                st.info(f'File Read! Data Extracted from {DATA_COLUMN}')

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
                        DATA = DATA.astype(str)
                        DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                        st.info(f'File Read! Data Extracted from {DATA_COLUMN}')
                except Exception as ex:
                    st.error(f'Error: {ex}. Try again.')

        elif CSP == 'Amazon':
            aws = csp_downloaders.AWSDownloader()
            if aws.SUCCESSFUL:
                try:
                    aws.downloadFile()
                    DATA = readFile(aws.AWS_FILE_NAME, MODE)
                    if not DATA.empty:
                        DATA = DATA.astype(str)
                        DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                        st.info(f'File Read! Data Extracted from {DATA_COLUMN}')
                except Exception as ex:
                    st.error(f'Error: {ex}. Try again.')

        elif CSP == 'Google':
            gcs = csp_downloaders.GoogleDownloader()
            if gcs.SUCCESSFUL:
                try:
                    gcs.downloadBlob()
                    DATA = readFile(gcs.GOOGLE_DESTINATION_FILE_NAME, MODE)
                    if not DATA.empty:
                        DATA = DATA.astype(str)
                        DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                        st.info(f'File Read! Data Extracted from {DATA_COLUMN}')
                except Exception as ex:
                    st.error(f'Error: {ex}. Try again.')

        elif CSP == 'Google Drive':
            gd = csp_downloaders.GoogleDriveDownloader()
            if gd.SUCCESSFUL:
                try:
                    gd.downloadBlob()
                    DATA = readFile(gd.GOOGLE_DRIVE_OUTPUT_FILENAME, MODE)
                    if not DATA.empty:
                        DATA = DATA.astype(str)
                        DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                        st.info(f'File Read! Data Extracted from {DATA_COLUMN}')
                except Exception as ex:
                    st.error(f'Error: {ex}. Try again.')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                              FUNCTION SELECTOR                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    st.markdown('# Analysis Functions\n'
                'Select the NLP functions which you wish to apply to your dataset.')
    APP_MODE = st.selectbox('Select the NLP Operation to execute',
                            ('Topic Modelling', 'Analyse Sentiment', 'Word Cloud', 'Named Entity Recognition',
                             'POS Tagging', 'Summarise'))

# -------------------------------------------------------------------------------------------------------------------- #
# |                                            WORD CLOUD VISUALISATION                                              | #
# -------------------------------------------------------------------------------------------------------------------- #
    if APP_MODE == 'Word Cloud':
        st.markdown('# Word Cloud Generation')

        # FLAGS
        st.markdown('## Flags')
        SAVE = st.checkbox('Output to image file?')
        MAX_WORDS = st.number_input('Key in the maximum number of words to display',
                                    min_value=2,
                                    max_value=1000,
                                    value=200)
        CONTOUR_WIDTH = st.number_input('Key in the contour width of your WordCloud',
                                        min_value=1,
                                        max_value=10,
                                        value=3)
        WIDTH = st.number_input('Key in the Width of the WordCloud image generated',
                                min_value=1,
                                max_value=100000,
                                value=800)
        HEIGHT = st.number_input('Key in the Width of the WordCloud image generated',
                                 min_value=1,
                                 max_value=100000,
                                 value=400)

        # MAIN DATA PROCESSING
        if st.button('Generate Word Cloud', key='wc'):
            DATA = DATA[[DATA_COLUMN]]
            wc = WordCloud(background_color='white',
                           max_words=MAX_WORDS,
                           contour_width=CONTOUR_WIDTH,
                           width=WIDTH,
                           height=HEIGHT,
                           contour_color='steelblue')
            wc.generate(' '.join(DATA[DATA_COLUMN]))

            st.markdown('## Wordcloud For Text Inputted')
            st.image(wc.to_image(), width=None)

            if SAVE:
                st.markdown('## Download Image')
                st.markdown('Download image from [downloads/wordcloud.png](downloads/wordcloud.png)')
                wc.to_file(str(DOWNLOAD_PATH / 'wordcloud.png'))


# -------------------------------------------------------------------------------------------------------------------- #
# |                                            NAMED ENTITY RECOGNITION                                              | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif APP_MODE == 'Named Entity Recognition':
        st.markdown('# Named Entity Recognition')
        st.markdown('Note that this module takes a long time to process a long piece of text. If you intend to process '
                    'large chunks of text, prepare to wait for hours for the NER Tagging process to finish. We are '
                    'looking to implement multiprocessing into the app to optimise it, but so far multiprocessing '
                    'does not seem to be fully supported in Streamlit.\n\n'
                    'In the meantime, it may be better to process your data in smaller batches to speed up your '
                    'workflow.')

        # FLAGS
        st.markdown('## Flags')
        st.markdown('Due to limits imposed on the visualisation engine and to avoid cluttering of the page with '
                    'outputs, you will only be able to visualise the NER outputs for a single piece of text at any '
                    'one point. However, you will still be able to download a text/html file containing '
                    'the outputs for you to save onto your disks.\n'
                    '### NLP Models\n'
                    'Select one model to use for your NLP Processing. Choose en_core_web_sm for a model that is '
                    'optimised for efficiency or en_core_web_lg for a model that is optimised for accuracy.')
        NLP_MODEL = st.radio('Select spaCy model', ('en_core_web_sm', 'en_core_web_lg'))
        if NLP_MODEL == 'en_core_web_sm':
            try:
                NLP = spacy.load('en_core_web_sm')
            except OSError:
                st.warning('Model not found, downloading...')
                try:
                    os.system('python -m spacy download en_core_web_sm')
                except Exception as ex:
                    st.error(f'Unable to download Model. Error: {ex}')
            except Exception as ex:
                st.error(f'Unknown Error: {ex}. Try again.')
            else:
                st.info('Efficiency Model Loaded!')
        elif NLP_MODEL == 'en_core_web_lg':
            try:
                NLP = spacy.load('en_core_web_lg')
            except OSError:
                st.warning('Model not found, downloading...')
                try:
                    os.system('python -m spacy download en_core_web_lg')
                except Exception as ex:
                    st.error(f'Unable to download Model. Error: {ex}')
            except Exception as ex:
                st.error(f'Unknown Error: {ex}. Try again.')
            else:
                st.info('Accuracy Model Loaded!')
        VERBOSE = st.checkbox('Display DataFrames?')
        SAVE = st.checkbox('Save Data?')
        if VERBOSE:
            VERBOSITY = st.slider('Choose Number of Data Points to Display (Select 0 to display all Data Points)',
                                  min_value=0,
                                  max_value=1000,
                                  value=20)
            ADVANCED_ANALYSIS = st.checkbox('Display Advanced DataFrame Statistics?')
        ONE_DATAPOINT = st.checkbox('Visualise One Data Point?')
        if ONE_DATAPOINT:
            DATAPOINT_SELECTOR = st.selectbox('Choose Data Point From Data', range(len(DATA)))
        else:
            st.info('You are conducting NER on the entire dataset. Only DataFrame is printed. NER output will be '
                    'automatically saved.')

        # MAIN PROCESSING
        if st.button('Conduct Named Entity Recognition', key='ner'):
            if not DATA.empty:
                # CLEAN UP AND STANDARDISE DATAFRAMES
                DATA = DATA[[DATA_COLUMN]]
                DATA['NER'] = ''
                DATA['COMPILED_LABELS'] = ''
                DATA = DATA.astype('str')

                for index in range(len(DATA)):
                    temp_nlp = NLP(DATA[DATA_COLUMN][index])
                    DATA.at[index, 'NER'] = str(list(zip([word.text for word in temp_nlp.ents],
                                                [word.label_ for word in temp_nlp.ents])))
                    DATA.at[index, 'COMPILED_LABELS'] = str(list(set([word.label_ for word in temp_nlp.ents])))

                if VERBOSE:
                    st.markdown('## NER DataFrame')
                    printDataFrame(data=DATA, verbose_level=VERBOSITY, advanced=ADVANCED_ANALYSIS)

                if ONE_DATAPOINT:
                    verbose_data_copy = DATA.copy()
                    temp_df = verbose_data_copy[DATA_COLUMN][DATAPOINT_SELECTOR]
                    st.markdown('## DisplaCy Rendering')
                    st.info('If rendering is not clean, choose to save files generated and download the rendering '
                            'in HTML format.')
                    SVG = displacy.render([sent for sent in NLP(str(temp_df)).sents],
                                          style='ent',
                                          page=True)
                    st.markdown(SVG, unsafe_allow_html=True)

                if SAVE:
                    st.markdown('## Download Data')
                    st.markdown('Download data from [downloads/ner.csv](downloads/ner.csv)')
                    DATA.to_csv(str(DOWNLOAD_PATH / 'ner.csv'), index=False)
                    if ONE_DATAPOINT:
                        st.markdown('Download data from [downloads/rendering.html](downloads/rendering.html)')
                        with open(pathlib.Path(str(DOWNLOAD_PATH / 'rendering.html')), 'w', encoding='utf-8') as f:
                            f.write(SVG)
            else:
                st.error('Error: Data not loaded properly. Try again.')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                             PART OF SPEECH TAGGING                                               | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif APP_MODE == 'POS Tagging':
        st.markdown('# POS Tagging')
        st.markdown('Note that this module takes a long time to process a long piece of text. If you intend to process '
                    'large chunks of text, prepare to wait for hours for the POS tagging process to finish. We are '
                    'looking to implement multiprocessing into the app to optimise it.\n\n'
                    'In the meantime, it may be better to process your data in smaller batches to speed up your '
                    'workflow.')

        # FLAGS
        st.markdown('## Flags')
        NLP_MODEL = st.radio('Select spaCy model', ('en_core_web_sm', 'en_core_web_lg'))
        if NLP_MODEL == 'en_core_web_sm':
            try:
                NLP = spacy.load('en_core_web_sm')
            except OSError:
                st.warning('Model not found, downloading...')
                try:
                    os.system('python -m spacy download en_core_web_sm')
                except Exception as ex:
                    st.error(f'Unable to download Model. Error: {ex}')
            except Exception as ex:
                st.error(f'Unknown Error: {ex}. Try again.')
            else:
                st.info('Efficiency Model Loaded!')
        elif NLP_MODEL == 'en_core_web_lg':
            try:
                NLP = spacy.load('en_core_web_lg')
            except OSError:
                st.warning('Model not found, downloading...')
                try:
                    os.system('python -m spacy download en_core_web_lg')
                except Exception as ex:
                    st.error(f'Unable to download Model. Error: {ex}')
            except Exception as ex:
                st.error(f'Unknown Error: {ex}. Try again.')
            else:
                st.info('Accuracy Model Loaded!')
        VERBOSE = st.checkbox('Display DataFrames?')
        if VERBOSE:
            VERBOSITY = st.slider('Choose Number of Data Points to Display (Select 0 to display all Data Points)',
                                  min_value=0,
                                  max_value=1000,
                                  value=20)
            ADVANCED_ANALYSIS = st.checkbox('Display Advanced DataFrame Statistics?')
        SAVE = st.checkbox('Save Output?')

        if st.button('Start POS Tagging'):
            if not DATA.empty:
                DATA['POS'] = np.nan
                DATA = DATA.astype(object)

                for index in range(len(DATA)):
                    temp_nlp = NLP(DATA[DATA_COLUMN][index])
                    DATA.at[index, 'POS'] = str(list(zip([str(word) for word in temp_nlp],
                                                         [word.pos_ for word in temp_nlp])))

                if VERBOSE:
                    st.markdown('## POS DataFrame')
                    printDataFrame(data=DATA, verbose_level=VERBOSITY, advanced=ADVANCED_ANALYSIS)

                if SAVE:
                    st.markdown('## Download Data')
                    st.markdown('Download data from [downloads/pos.csv](downloads/pos.csv)')
                    DATA.to_csv(str(DOWNLOAD_PATH / 'pos.csv'), index=False)

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                 SUMMARIZATION                                                    | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif APP_MODE == 'Summarise':
        st.markdown('# Summarization of Text')
        st.markdown('Note that this module takes a long time to process a long piece of text. If you intend to process '
                    'large chunks of text, prepare to wait for hours for the summarization process to finish. We are '
                    'looking to implement multiprocessing into the app to optimise it.\n\n'
                    'In the meantime, it may be better to process your data in smaller batches to speed up your '
                    'workflow.')

        # FLAGS
        st.markdown('## Flags')
        st.markdown('Select one model to use for your NLP Processing. Choose en_core_web_sm for a model that is '
                    'optimised for efficiency or en_core_web_lg for a model that is optimised for accuracy.')
        NLP_MODEL = st.radio('Select spaCy model', ('en_core_web_sm', 'en_core_web_lg'))
        if NLP_MODEL == 'en_core_web_sm':
            try:
                NLP = spacy.load('en_core_web_sm')
            except OSError:
                st.warning('Model not found, downloading...')
                try:
                    os.system('python -m spacy download en_core_web_sm')
                except Exception as ex:
                    st.error(f'Unable to download Model. Error: {ex}')
            except Exception as ex:
                st.error(f'Unknown Error: {ex}. Try again.')
            else:
                st.info('Efficiency Model Loaded!')
        elif NLP_MODEL == 'en_core_web_lg':
            try:
                NLP = spacy.load('en_core_web_lg')
            except OSError:
                st.warning('Model not found, downloading...')
                try:
                    os.system('python -m spacy download en_core_web_lg')
                except Exception as ex:
                    st.error(f'Unable to download Model. Error: {ex}')
            except Exception as ex:
                st.error(f'Unknown Error: {ex}. Try again.')
            else:
                st.info('Accuracy Model Loaded!')
        SENT_LEN = st.number_input('Enter the total number of sentences to summarise text to',
                                   min_value=1,
                                   max_value=100,
                                   value=3)
        SAVE = st.checkbox('Output to CSV file?')
        VERBOSE = st.checkbox('Print out DataFrames?')
        if VERBOSE:
            VERBOSITY = st.slider('Data points',
                                  key='Data points to display?',
                                  min_value=1,
                                  max_value=1000,
                                  value=20)
            ADVANCED_ANALYSIS = st.checkbox('Display Advanced DataFrame Statistics?')

        # MAIN PROCESSING
        if st.button('Summarise Text', key='runner'):
            try:
                # CLEAN UP AND STANDARDISE DATAFRAMES
                DATA = DATA[[DATA_COLUMN]]
                DATA['SUMMARY'] = np.nan
                DATA = DATA.astype(str)
            except KeyError:
                st.error('Warning: CLEANED CONTENT is not found in the file uploaded. Try again.')
            except Exception as ex:
                st.error(ex)
            else:
                stopwords = list(STOP_WORDS)
                pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
                DATA['SUMMARY'] = DATA[DATA_COLUMN].apply(lambda x: summarise(x, stopwords, pos_tag, NLP, SENT_LEN))

            # SHOW DATASETS
            if VERBOSE:
                st.markdown('## Summary DataFrame')
                printDataFrame(data=DATA, verbose_level=VERBOSITY, advanced=ADVANCED_ANALYSIS)

            # SAVE DATA
            if SAVE:
                st.markdown('## Download Summarised Data')
                st.markdown('Download summarised data from [downloads/summarised.csv]'
                            '(downloads/summarised.csv)')
                DATA.to_csv(str(DOWNLOAD_PATH / 'summarised.csv'), index=False)
            else:
                st.error('Warning: Data is processed wrongly. Try again.')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                              SENTIMENT ANALYSIS                                                  | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif APP_MODE == 'Analyse Sentiment':
        st.markdown('# Sentiment Analysis')

        # FLAGS
        st.markdown('## Flags')
        BACKEND_ANALYSER = st.selectbox('Choose the backend engine used to conduct sentiment analysis on your text',
                                        ('VADER', 'TextBlob'))
        SAVE = st.checkbox('Output to CSV file?')
        VERBOSE = st.checkbox('Print out the outputs to screen?')
        if VERBOSE:
            VERBOSITY = st.slider('Data points', key='Data points to display?', min_value=1, max_value=1000, value=20)
            COLOUR = st.color_picker('Choose Colour of Marker to Display', value='#2ACAEA')
            ADVANCED_ANALYSIS = st.checkbox('Display Advanced DataFrame Statistics?')

        # MAIN PROCESSING
        if st.button('Start Analysis', key='analysis'):
            if BACKEND_ANALYSER == 'VADER':
                replacer = {
                    r"'": '',
                    r'[^\w\s]': ' ',
                    r' \d+': ' ',
                    r' +': ' '
                }

                DATA['VADER SENTIMENT TEXT'] = DATA[DATA_COLUMN].replace(to_replace=replacer, regex=True)

                vader_analyser = SentimentIntensityAnalyzer()
                sent_score_list = list()
                sent_label_list = list()

                for i in DATA['VADER SENTIMENT TEXT'].values.tolist():
                    sent_score = vader_analyser.polarity_scores(i)

                    if sent_score['compound'] >= 0.05:
                        sent_score_list.append(sent_score['compound'])
                        sent_label_list.append('Positive')
                    elif -0.05 < sent_score['compound'] < 0.05:
                        sent_score_list.append(sent_score['compound'])
                        sent_label_list.append('Neutral')
                    elif sent_score['compound'] <= -0.05:
                        sent_score_list.append(sent_score['compound'])
                        sent_label_list.append('Negative')

                DATA['VADER SENTIMENT'] = sent_label_list
                DATA['VADER SCORE'] = sent_score_list

            elif BACKEND_ANALYSER == 'TextBlob':
                if not DATA.empty:
                    # APPLY LAMBDAS ON THE DATAFRAME
                    DATA['POLARITY'] = DATA[DATA_COLUMN].apply(lambda x: TextBlob(x).sentiment.polarity)
                    DATA['SUBJECTIVITY'] = DATA[DATA_COLUMN].apply(lambda x: TextBlob(x).sentiment.subjectivity)
                else:
                    st.error('Error: Data file not loaded properly. Try again.')

            # SHOW DATA
            if VERBOSE:
                if BACKEND_ANALYSER == 'VADER':
                    if 'VADER SENTIMENT' or 'VADER SCORE' in DATA.columns:
                        st.markdown('## Sentiment DataFrame')
                        printDataFrame(data=DATA, verbose_level=VERBOSITY, advanced=ADVANCED_ANALYSIS)

                        st.markdown('## Kernel Density Plot')
                        HAC_PLOT = ff.create_distplot([DATA['VADER SCORE'].tolist()], ['VADER'],
                                                      colors=[COLOUR], show_hist=False, show_rug=False)
                        HAC_PLOT.update_layout(title_text='Histogram and Curve Plot',
                                               xaxis_title='VADER Score',
                                               yaxis_title='Frequency Density',
                                               legend_title='Frequency Density',
                                               )
                        st.plotly_chart(HAC_PLOT)
                    else:
                        st.error('Warning: An error is made in the processing of the data. Try again.')

                elif BACKEND_ANALYSER == 'TextBlob':
                    if 'POLARITY' or 'SUBJECTIVITY' in DATA.columns:
                        st.markdown('## Sentiment DataFrame')
                        printDataFrame(data=DATA, verbose_level=VERBOSITY, advanced=ADVANCED_ANALYSIS)

                        st.markdown('## Visualise Polarity VS Subjectivity')
                        HAC_PLOT = px.scatter(DATA[['SUBJECTIVITY', 'POLARITY']],
                                              x='SUBJECTIVITY',
                                              y='POLARITY',
                                              labels={
                                                  'SUBJECTIVITY': 'Subjectivity',
                                                  'POLARITY': 'Polarity'
                                              })
                        st.plotly_chart(HAC_PLOT, use_container_width=True)
                    else:
                        st.error('Warning: An error is made in the processing of the data. Try again.')

            # SAVE DATA
            if SAVE:
                st.markdown('## Download Data')
                st.markdown('Download sentiment data from [downloads/sentiment_scores.csv]'
                            '(downloads/sentiment_scores.csv)')
                DATA.to_csv(str(DOWNLOAD_PATH / "sentiment_scores.csv"), index=False)

                if HAC_PLOT is not None:
                    st.markdown('## Graph')
                    st.markdown('Download sentiment data from [downloads/plot.png](downloads/plot.png)')
                    HAC_PLOT.write_image(str(DOWNLOAD_PATH / 'plot.png'))

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                TOPIC MODELLING                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif APP_MODE == 'Topic Modelling':
        st.markdown('# Topic Modelling')
        st.markdown('Ensure that your data is **lemmatized and properly cleaned**; data **should not be tokenized** '
                    'for this step. Use the Load, Clean and Visualise module to clean and lemmatize your data if you '
                    'have not done so already.\n\n'
                    '## Short Explanation on Models used\n'
                    '### Latent Dirichlet Allocation (LDA)\n'
                    'Extracted from '
                    'https://towardsdatascience.com/topic-modeling-quora-questions-with-lda-nmf-aff8dce5e1dd, LDA is '
                    'the Model used for finding all the hidden probability distributions of a certain set of data, '
                    'which would allow us to discover hidden groupings of data points within a set of data.\n\n'
                    '### Non-Negative Matrix Factorization (NMF)\n'
                    'Extracted from '
                    'https://medium.com/analytics-vidhya/topic-modeling-with-non-negative-matrix-factorization-nmf'
                    '-3caf3a6bb6da, NMF is an unsupervised learning technique to decompose (factorise) high-'
                    'dimensional vectors into non-negative lower-dimensional vectors.\n\n'
                    '### Latent Semantic Indexing (LSI)\n'
                    'Extracted from https://www.searchenginejournal.com/latent-semantic-indexing-wont-help-seo/240705/'
                    '#:~:text=Latent%20semantic%20indexing%20(also%20referred,of%20those%20words%20and%20documents, '
                    'LSI is a technique of analysing a document to discover statistical co-occurrences of words which '
                    'appear to gather, which gives us insights into the topics of the words and of the document.')

        # FLAGS
        st.markdown('## Flags')
        SAVE = st.checkbox('Output to file?')
        VERBOSE = st.checkbox('Display Outputs?')
        if VERBOSE:
            VERBOSITY = st.slider('Data points', key='Data points to display?', min_value=1, max_value=1000, value=20)
            ADVANCED_ANALYSIS = st.checkbox('Display Advanced DataFrame Statistics?')
        NLP_TOPIC_MODEL = st.selectbox('Choose Model to use', ('Latent Dirichlet Allocation',
                                                               'Non-Negative Matrix Factorization',
                                                               'Latent Semantic Indexing'))
        NUM_TOPICS = st.number_input('Choose number of topics to generate per text',
                                     min_value=1,
                                     max_value=100,
                                     step=1,
                                     value=10)
        MAX_FEATURES = st.number_input('Choose the maximum number of features to extract from dataset',
                                       min_value=1,
                                       max_value=99999,
                                       step=1,
                                       value=5000)
        MAX_ITER = st.number_input('Choose number of iterations of model training',
                                   min_value=1,
                                   max_value=100,
                                   step=1,
                                   value=10)

        if NLP_TOPIC_MODEL == 'Latent Dirichlet Allocation':
            MIN_DF = st.number_input('Choose minimum (cardinal) frequency of words to consider',
                                     min_value=1,
                                     max_value=100,
                                     step=1,
                                     value=5)
            MAX_DF = st.number_input('Choose maximum (percentage) frequency of words to consider',
                                     min_value=0.,
                                     max_value=1.,
                                     step=0.01,
                                     format='%.2f',
                                     value=0.90)
            WORKER = st.number_input('Chose number of CPU Cores to use',
                                     min_value=1,
                                     max_value=multiprocessing.cpu_count(),
                                     step=1,
                                     value=1)
        elif NLP_TOPIC_MODEL == 'Non-Negative Matrix Factorization':
            MIN_DF = st.number_input('Choose minimum (cardinal) frequency of words to consider',
                                     min_value=1,
                                     max_value=100,
                                     step=1,
                                     value=2)
            MAX_DF = st.number_input('Choose maximum (percentage) frequency of words to consider',
                                     min_value=0.,
                                     max_value=1.,
                                     step=0.01,
                                     format='%.2f',
                                     value=0.95)
            ALPHA = st.number_input('Choose alpha value for NMF Model',
                                    min_value=0.,
                                    max_value=1.,
                                    step=0.01,
                                    format='%.2f',
                                    value=.1)
            L1_RATIO = st.number_input('Choose L1 Ratio for NMF Model',
                                       min_value=0.,
                                       max_value=1.,
                                       step=0.01,
                                       format='%.2f',
                                       value=.5)
        elif NLP_TOPIC_MODEL == 'Latent Semantic Indexing':
            if VERBOSE:
                COLOUR = st.color_picker('Choose Colour of Marker to Display', value='#2ACAEA')

        if st.button('Start Modelling', key='topic'):
            if not DATA.empty:
                CV = CountVectorizer(min_df=MIN_DF,
                                     max_df=MAX_DF,
                                     stop_words='english',
                                     lowercase=True,
                                     token_pattern=r'[a-zA-Z\-][a-zA-Z\-]{2,}',
                                     max_features=MAX_FEATURES)
                try:
                    VECTORISED = CV.fit_transform(DATA[DATA_COLUMN])
                except ValueError:
                    st.error('Error: The column loaded is empty or has invalid data points. Try again.')

                # LDA
                if NLP_TOPIC_MODEL == 'Latent Dirichlet Allocation':
                    LDA_MODEL = LatentDirichletAllocation(n_components=NUM_TOPICS,
                                                          max_iter=MAX_ITER,
                                                          learning_method='online',
                                                          n_jobs=WORKER)
                    LDA_DATA = LDA_MODEL.fit_transform(VECTORISED)

                    st.markdown('## Model Data')
                    TOPIC_TEXT = modelIterator(LDA_MODEL, CV, top_n=NUM_TOPICS)

                    # GET DOMINANT TOPICS
                    DATA['DOMINANT TOPIC'] = DATA[DATA_COLUMN].apply(lambda x:
                                                                     (LDA_MODEL.transform(CV.transform([x]))[0]))

                    KW = pd.DataFrame(dominantTopic(vect=CV, model=LDA_MODEL, n_words=NUM_TOPICS))
                    KW.columns = [f'word_{i}' for i in range(KW.shape[1])]
                    KW.columns = [f'topic_{i}' for i in range(KW.shape[0])]

                    # THIS VISUALISES ALL THE DOCUMENTS IN THE DATASET PROVIDED
                    LDA_VIS = pyLDAvis.sklearn.prepare(LDA_MODEL, VECTORISED, CV, mds='tsne')

                    if VERBOSE:
                        st.markdown('## DataFrame')
                        printDataFrame(data=DATA, verbose_level=VERBOSITY, advanced=ADVANCED_ANALYSIS)

                        st.markdown('## Topic Label\n'
                                    'The following frame will show you the words that are associated with a certain '
                                    'topic.')
                        printDataFrame(data=KW, verbose_level=NUM_TOPICS, advanced=False)

                        st.markdown('## LDA\n'
                                    f'The following HTML render displays the top {NUM_TOPICS} of Topics generated '
                                    f'from all the text provided in your dataset.')
                        LDA_VIS_STR = pyLDAvis.prepared_data_to_html(LDA_VIS)
                        streamlit.components.v1.html(LDA_VIS_STR, width=1300, height=800)

                    if SAVE:
                        st.markdown('## Save Data\n'
                                    '### Topics')
                        for i in range(len(TOPIC_TEXT)):
                            st.markdown(f'Download all Topic List from [downloads/lda_topics_{i}.csv]'
                                        f'(downloads/lda_topics_{i}.csv)')
                            TOPIC_TEXT[i].to_csv(str(DOWNLOAD_PATH / f'lda_topics_{i}.csv'), index=False)
                        st.markdown('### Topic/Word List')
                        st.markdown(f'Download Summarised Topic/Word List from [downloads/summary_topics.csv]'
                                    f'(downloads/summary_topics.csv)')
                        KW.to_csv(str(DOWNLOAD_PATH / 'summary_topics.csv'))

                        st.markdown('### Other Requested Data')
                        st.markdown('Download HTML File from [downloads/lda.html](downloads/lda.html)')
                        pyLDAvis.save_html(LDA_VIS, str(DOWNLOAD_PATH / 'lda.html'))

                # NMF
                elif NLP_TOPIC_MODEL == 'Non-Negative Matrix Factorization':
                    TFIDF_MODEL = TfidfVectorizer(max_df=MAX_DF,
                                                  min_df=MIN_DF,
                                                  max_features=MAX_FEATURES,
                                                  stop_words='english')
                    TFIDF_VECTORISED = TFIDF_MODEL.fit_transform(DATA[DATA_COLUMN].values.astype(str))
                    NMF_MODEL = NMF(n_components=NUM_TOPICS,
                                    max_iter=MAX_ITER,
                                    random_state=1,
                                    alpha=ALPHA,
                                    l1_ratio=L1_RATIO).fit(TFIDF_VECTORISED)
                    NMF_DATA = NMF_MODEL.fit_transform(TFIDF_VECTORISED)

                    st.markdown('## Model Data')
                    TOPIC_TEXT = modelNMFIterator(NMF_MODEL, TFIDF_MODEL, top_n=NUM_TOPICS)
                    TOPIC_FRAME = pd.DataFrame(TOPIC_TEXT)

                    KW = pd.DataFrame(TOPIC_TEXT)
                    KW.columns = [f'word_{i}' for i in range(KW.shape[1])]
                    KW.columns = [f'topic_{i}' for i in range(KW.shape[0])]

                    if VERBOSE:
                        st.markdown('## NMF Data')
                        printDataFrame(data=DATA, verbose_level=VERBOSITY, advanced=ADVANCED_ANALYSIS)
                        st.markdown('## NMF Topic DataFrame')
                        printDataFrame(data=KW, verbose_level=VERBOSITY, advanced=ADVANCED_ANALYSIS)

                    if SAVE:
                        st.markdown('## Save Data\n'
                                    '### Topics')
                        for i in range(len(TOPIC_TEXT)):
                            st.markdown(f'Download Topic List from [downloads/nmf_topics_{i}.csv]'
                                        f'(downloads/nmf_topics_{i}.csv)')
                            TOPIC_FRAME[i].to_csv(str(DOWNLOAD_PATH / f'nmf_topics_{i}.csv'), index=False)
                        st.markdown('### Topic/Word List')
                        st.markdown(f'Download Summarised Topic/Word List from [downloads/summary_topics.csv]'
                                    f'(downloads/summary_topics.csv)')
                        st.markdown('Download Processed Data from [downloads/processed.csv](downloads/processed.csv)')
                        DATA.to_csv(str(DOWNLOAD_PATH / 'processed.csv'))

                # LSI
                elif NLP_TOPIC_MODEL == 'Latent Semantic Indexing':
                    LSI_MODEL = TruncatedSVD(n_components=NUM_TOPICS,
                                             n_iter=MAX_ITER)
                    LSI_DATA = LSI_MODEL.fit_transform(VECTORISED)

                    st.markdown('## Model Data')
                    TOPIC_TEXT = modelIterator(LSI_MODEL, CV, top_n=NUM_TOPICS)

                    if VERBOSE:
                        st.markdown('## LSI(SVD) Scatterplot\n'
                                    'Note that the following visualisation will use a Topic count of 2, '
                                    'overriding previous inputs above, as you are only able to visualise data '
                                    'on 2 axis. However the full analysis result of the above operation will '
                                    'be saved in the dataset you provided and be available for download later '
                                    'on in the app.\n\n'
                                    'The main aim of the scatterplot is to show the similarity between topics, '
                                    'which is measured by the distance between markers as shown in the following '
                                    'diagram. The diagram contained within the expander is the same as the marker '
                                    'diagram, just that the markers are all replaced by the topic words the '
                                    'markers actually represent.')
                        svd_2d = TruncatedSVD(n_components=2)
                        data_2d = svd_2d.fit_transform(VECTORISED)

                        MAR_FIG = go.Scattergl(
                            x=data_2d[:, 0],
                            y=data_2d[:, 1],
                            mode='markers',
                            marker=dict(
                                color=COLOUR,
                                line=dict(width=1)
                            ),
                            text=CV.get_feature_names(),
                            hovertext=CV.get_feature_names(),
                            hoverinfo='text'
                        )
                        MAR_FIG = [MAR_FIG]
                        MAR_FIG = go.Figure(data=MAR_FIG)
                        st.plotly_chart(MAR_FIG)

                        with st.expander('Show Word Plots'):
                            WORD_FIG = go.Scattergl(
                                x=data_2d[:, 0],
                                y=data_2d[:, 1],
                                mode='text',
                                marker=dict(
                                    color=COLOUR,
                                    line=dict(width=1)
                                ),
                                text=CV.get_feature_names(),
                            )
                            WORD_FIG = [WORD_FIG]
                            WORD_FIG = go.Figure(data=WORD_FIG)
                            st.plotly_chart(WORD_FIG)

                        if SAVE:
                            st.markdown('## Save Data\n'
                                        '### Topics')
                            for i in range(len(TOPIC_TEXT)):
                                st.markdown(f'Download Topic List from [downloads/lsi_topic_{i}.csv]'
                                            f'(downloads/lsi_topic_{i}.csv)')
                                TOPIC_TEXT[i].to_csv(str(DOWNLOAD_PATH / f'lsi_topic_{i}.csv'), index=False)

                            st.markdown('### Other Requested Data')
                            st.markdown('Download PNG File from [downloads/marker_figure.png]'
                                        '(downloads/marker_figure.png)')
                            MAR_FIG.write_image(str(DOWNLOAD_PATH / 'marker_figure.png'))
                            st.markdown('Download PNG File from [downloads/word_figure.png]'
                                        '(downloads/word_figure.png)')
                            WORD_FIG.write_image(str(DOWNLOAD_PATH / 'word_figure.png'))
            else:
                st.error('Error: File not loaded properly. Try again.')
