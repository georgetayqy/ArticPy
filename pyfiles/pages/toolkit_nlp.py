"""
This module allows the user to conduct basic NLP analysis functions on the preprocessed data using the spaCy module

This module uses CPU-optimised pipelines and hence a GPU is optional in this module
"""

# -------------------------------------------------------------------------------------------------------------------- #
# |                                         IMPORT RELEVANT LIBRARIES                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
import os
import pathlib

import numpy as np
import pandas as pd
import spacy
import streamlit as st
import plotly.express as px
import nltk
import gensim
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models
import pandas_profiling

from streamlit_pandas_profiling import st_profile_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from spacy.lang.en.stop_words import STOP_WORDS
from spacy import displacy
from wordcloud import WordCloud
from textblob import TextBlob
from utils import csp_downloaders
from utils.helper import readFile, summarise, sent2word, stopwordRemover

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
SENT_LEN = 3
NUM_TOPICS = 10
TOPIC_LDA = pd.DataFrame()
TOPIC_LDA_STR = pd.DataFrame()
LDA_DATA = pd.DataFrame()
LDA_VIS = None
LDA_MODEL = None
DOCUMENT_ID = 0
MODEL = None
# NOTE THAT DOCUMENT_ID IS ZERO-INDEXED IN THIS CASE, BUT IT CORRESPONDS TO 1 ON THE DATASET
FINALISED_DATA_LIST = []
ADVANCED_ANALYSIS = False
NLP_MODEL = 'en_core_web_sm'
DATA_COLUMN = None
NLP = None
ONE_DATAPOINT = False
DATAPOINT_SELECTOR = 0


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
        CONTOUR_WIDTH, DATA, SENT_LEN, NUM_TOPICS, LDA_MODEL, MODEL, DOCUMENT_ID, TOPIC_LDA, TOPIC_LDA_STR, \
        FINALISED_DATA_LIST, LDA_DATA, LDA_VIS, ADVANCED_ANALYSIS, NLP_MODEL, DATA_COLUMN, NLP, ONE_DATAPOINT, \
        DATAPOINT_SELECTOR

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                    INIT                                                          | #
# -------------------------------------------------------------------------------------------------------------------- #
    # CREATE THE DOWNLOAD PATH WHERE FILES CREATED WILL BE STORED AND AVAILABLE FOR DOWNLOADING
    if not DOWNLOAD_PATH.is_dir():
        DOWNLOAD_PATH.mkdir()

    st.title('NLP Toolkit')
    st.markdown('## Init\n'
                'This module uses the gensim and spaCy package to conduct the necessary NLP preprocessing and '
                'analysis tasks for users to make sense of the text they pass into app. Note that this app requires '
                'the data to be decently cleaned; if you have not done so, run the Load, Clean adn Visualise module '
                'and save the cleaned and tokenized data onto your workstation. Those files may come in useful in '
                'the functionality of this app.\n\n')
    st.markdown('## Upload Data\n'
                'Due to limitations imposed by the file uploader widget, only files smaller than 200 MB can be loaded '
                'with the widget. If your files are larger than 200 MB, please select "Large File(s)" and select the '
                'CSP you are using to host and store your data. Next, select the file format you wish to upload.')
    FILE = st.selectbox('Select the type of file to load', ('Small File(s)', 'Large File(s)'))
    MODE = st.selectbox('Define the data input format', ('CSV', 'XLSX'))

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
            if st.button('Read File', key='az'):
                if azure.SUCCESSFUL:
                    try:
                        azure.downloadBlob()
                        DATA = readFile(azure.AZURE_DOWNLOAD_ABS_PATH, MODE)
                        if not DATA.empty:
                            DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                            st.info('File Read!')
                    except AttributeError:
                        st.error(f'Error: {AttributeError}, one or more parameters are not loaded properly. Try again.')
                else:
                    st.error('Error: Parameters are not loaded or is validated successfully. Try again.')

        elif CSP == 'Amazon':
            aws = csp_downloaders.AWSDownloader()
            if st.button('Read File', key='aws'):
                if aws.SUCCESSFUL:
                    try:
                        aws.downloadFile()
                        DATA = readFile(aws.AWS_FILE_NAME, MODE)
                        if not DATA.empty:
                            DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                            st.info('File Read!')
                    except AttributeError:
                        st.error(f'Error: {AttributeError}, one or more parameters are not loaded properly. Try again.')
                else:
                    st.error('Error: Parameters are not loaded or is validated successfully. Try again.')

        elif CSP == 'Google':
            gcs = csp_downloaders.GoogleDownloader()
            if st.button('Read File', key='gcs'):
                if gcs.SUCCESSFUL:
                    try:
                        gcs.downloadBlob()
                        DATA = readFile(gcs.GOOGLE_DESTINATION_FILE_NAME, MODE)
                        if not DATA.empty:
                            DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                            st.info('File Read!')
                    except AttributeError:
                        st.error(f'Error: {AttributeError}, one or more parameters are not loaded properly. Try again.')
                else:
                    st.error('Error: Parameters are not loaded or is validated successfully. Try again.')

        elif CSP == 'Google Drive':
            gd = csp_downloaders.GoogleDriveDownloader()
            if st.button('Read File', key='gd'):
                if gd.SUCCESSFUL:
                    try:
                        gd.downloadBlob()
                        DATA = readFile(gd.GOOGLE_DRIVE_OUTPUT_FILENAME, MODE)
                        if not DATA.empty:
                            DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                            st.info('File Read!')
                    except AttributeError:
                        st.error(f'Error: {AttributeError}, one or more parameters are not loaded properly. Try again.')
                else:
                    st.error('Error: Parameters are not loaded or is validated successfully. Try again.')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                              FUNCTION SELECTOR                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    st.markdown('## Analysis Functions\n'
                'Select the NLP functions which you wish to apply to your dataset.')
    APP_MODE = st.selectbox('Select the NLP Operation to execute',
                            ('Word Cloud', 'Named Entity Recognition', 'Summarise', 'Analyse Sentiment',
                             'Topic Modelling'))

# -------------------------------------------------------------------------------------------------------------------- #
# |                                            WORD CLOUD VISUALISATION                                              | #
# -------------------------------------------------------------------------------------------------------------------- #
    if APP_MODE == 'Word Cloud':
        st.markdown('# Word Cloud Generation')
# -------------------------------------------------------------------------------------------------------------------- #
# |                                               PROCESSING FLAGS                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
        st.markdown('## Flags')
        SAVE = st.checkbox('Output to image file?')
        MAX_WORDS = st.number_input('Key in the maximum number of words to display',
                                    min_value=2,
                                    max_value=1000,
                                    value=200)
        CONTOUR_WIDTH = st.number_input('Key in the contour width of your wordcloud',
                                        min_value=1,
                                        max_value=10,
                                        value=3)

# -------------------------------------------------------------------------------------------------------------------- #
# |                                           DATA LOADING AND PROCESSING                                            | #
# -------------------------------------------------------------------------------------------------------------------- #
        if st.button('Generate Word Cloud', key='wc'):
            DATA = DATA[[DATA_COLUMN]]
            temp_list = DATA[DATA_COLUMN].tolist()

            temp_text = ' '
            for i in range(len(temp_list)):
                temp_text = str(temp_list[i]) + temp_text

            wc = WordCloud(background_color='white',
                           max_words=MAX_WORDS,
                           contour_width=CONTOUR_WIDTH,
                           contour_color='steelblue')
            wc.generate(temp_text)

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
# -------------------------------------------------------------------------------------------------------------------- #
# |                                               PROCESSING FLAGS                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
        st.markdown('## Flags')
        st.markdown('Due to limits imposed on the visualisation engine and to avoid cluttering of the page with '
                    'outputs, you will only be able to visualise the NER outputs for a single piece of text at any '
                    'one point. However, you will still be able to download a text/html file containing '
                    'the outputs for you to save onto your disks.')
        st.markdown('### NLP Models')
        st.markdown('Select one model to use for your NLP Processing. Choose en_core_web_sm for a model that is '
                    'optimised for efficiency or en_core_web_lg for a model that is optimised for accuracy.')
        NLP_MODEL = st.radio('Select spaCy model', ('en_core_web_sm', 'en_core_web_lg'))
        if NLP_MODEL == 'en_core_web_sm':
            try:
                os.system('python -m spacy download en_core_web_sm')
            except Exception as ex:
                st.error(f'Error: {ex}')
            else:
                st.info('Efficiency Model Downloaded!')
        elif NLP_MODEL == 'en_core_web_lg':
            try:
                os.system('python -m spacy download en_core_web_lg')
            except Exception as ex:
                st.error(f'Error: {ex}')
            else:
                st.info('Accuracy Model Downloaded!')
        VERBOSE = st.checkbox('Display DataFrames?')
        if VERBOSE:
            VERBOSITY = st.slider('Choose Number of Data Points to Display (Select 0 to display all Data Points)',
                                  min_value=0,
                                  max_value=1000,
                                  value=20)
        ONE_DATAPOINT = st.checkbox('Visualise One Data Point?')
        if ONE_DATAPOINT:
            DATAPOINT_SELECTOR = st.selectbox('Choose Data Point From Data', range(len(DATA)))
            SAVE = st.checkbox('Analyse Entire Dataset and Save NER Output?')
        else:
            SAVE = True
            QUERY = st.checkbox('Conduct Query for Explanation for the Labels for Text?')
            if QUERY:
                DATAPOINT_SELECTOR = st.selectbox('Choose Data Point From Data', range(len(DATA)))
            st.info('You are conducting NER on the entire dataset. Only DataFrame is printed. NER output will be '
                    'automatically saved.')


# -------------------------------------------------------------------------------------------------------------------- #
# |                                           DATA LOADING AND PROCESSING                                            | #
# -------------------------------------------------------------------------------------------------------------------- #
        if st.button('Conduct Named Entity Recognition', key='ner'):
            try:
                # CLEAN UP AND STANDARDISE DATAFRAMES
                DATA = DATA[[DATA_COLUMN]]
                DATA = DATA.astype(str)
            except KeyError:
                st.error('Warning: CLEANED CONTENT is not found in the file uploaded. Try again.')

            if not DATA.empty:
                if NLP_MODEL == 'en_core_web_sm':
                    NLP = spacy.load('en_core_web_sm')
                elif NLP_MODEL == 'en_core_web_lg':
                    NLP = spacy.load('en_core_web_lg')

                for index, row in DATA.iterrows():
                    temp_word_list = []
                    temp_word_label = []
                    temp_nlp = NLP(row[DATA_COLUMN])
                    for word in temp_nlp.ents:
                        temp_word_list.append(word.text)
                        temp_word_label.append(word.label_)
                    DATA.at[index, 'NER'] = list(zip(temp_word_list, temp_word_list))
                    DATA.at[index, 'COMPILED_LABELS'] = set(temp_word_label)

                if VERBOSE:
                    if VERBOSITY != 0:
                        try:
                            if not DATA.empty:
                                st.markdown('## DataFrame')
                                st.dataframe(DATA.head(VERBOSITY), height=400, width=800)

                                if ADVANCED_ANALYSIS:
                                    with st.expander('Advanced Profile Report'):
                                        st_profile_report(DATA.profile_report(
                                            explorative=True,
                                            minimal=True
                                        ))
                        except RuntimeError:
                            st.warning('Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                            st.markdown('## DataFrame')
                            st.dataframe(DATA.head(10), height=400, width=800)

                            if ADVANCED_ANALYSIS:
                                with st.expander('Advanced Profile Report'):
                                    st_profile_report(DATA.profile_report(
                                        explorative=True,
                                        minimal=True
                                    ))
                        except KeyError:
                            st.error('Warning: Your data was not processed properly. Try again.')

                    else:
                        try:
                            if not DATA.empty:
                                st.markdown('## DataFrame')
                                st.dataframe(DATA, height=400, width=800)

                                if ADVANCED_ANALYSIS:
                                    with st.expander('Advanced Profile Report'):
                                        st_profile_report(DATA.profile_report(
                                            explorative=True,
                                            minimal=True
                                        ))
                        except RuntimeError:
                            st.warning('Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                            st.markdown('## DataFrame')
                            st.dataframe(DATA.head(10), height=400, width=800)

                            if ADVANCED_ANALYSIS:
                                with st.expander('Advanced Profile Report'):
                                    st_profile_report(DATA.profile_report(
                                        explorative=True,
                                        minimal=True
                                    ))
                        except KeyError:
                            st.error('Warning: Your data was not processed properly. Try again.')

                if ONE_DATAPOINT:
                    verbose_data_copy = DATA.copy()
                    temp_df = verbose_data_copy.iloc[int(DATAPOINT_SELECTOR)]
                    st.markdown('## DisplaCy Rendering')
                    st.markdown(displacy.render(temp_df['PROCESSED_TEXT'], style="ent"), unsafe_allow_html=True)
                    with st.expander('Get Explanation for Label'):
                        choice = st.selectbox('Select Label to Explain', verbose_data_copy['COMPILED_LABELS'])
                        st.info(str(spacy.explain(choice)))

                    if SAVE:
                        st.markdown('## Download Data')
                        st.markdown('Download data from [downloads/ner.csv](downloads/ner.csv)')
                        DATA.to_csv(str(DOWNLOAD_PATH / 'ner.csv'))
                else:
                    st.markdown('## Query and Explanation')
                    data_copy = DATA.copy()
                    temp_df = data_copy.iloc[DATAPOINT_SELECTOR]
                    st.markdown(displacy.render(temp_df['PROCESSED_TEXT'], style="ent"), unsafe_allow_html=True)
                    with st.expander('Get Explanation for Label'):
                        choice = st.selectbox('Select Label to Explain', data_copy['COMPILED_LABELS'])
                        st.info(str(spacy.explain(choice)))

                    if SAVE:
                        st.markdown('## Download Data')
                        st.markdown('Download data from [downloads/ner.csv](downloads/ner.csv)')
                        DATA.to_csv(str(DOWNLOAD_PATH / 'ner.csv'))


# -------------------------------------------------------------------------------------------------------------------- #
# |                                                 SUMMARIZATION                                                    | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif APP_MODE == 'Summarise':
        st.markdown('# Summarization of Text')
# -------------------------------------------------------------------------------------------------------------------- #
# |                                                PROCESSING FLAGS                                                  | #
# -------------------------------------------------------------------------------------------------------------------- #
        st.markdown('## Flags')
        st.markdown('Select one model to use for your NLP Processing. Choose en_core_web_sm for a model that is '
                    'optimised for efficiency or en_core_web_lg for a model that is optimised for accuracy.')
        NLP_MODEL = st.radio('Select spaCy model', ('en_core_web_sm', 'en_core_web_lg'))
        if NLP_MODEL == 'en_core_web_sm':
            try:
                os.system('python -m spacy download en_core_web_sm')
            except Exception as ex:
                st.error(f'Error: {ex}')
            else:
                st.info('Efficiency Model Downloaded!')
        elif NLP_MODEL == 'en_core_web_lg':
            try:
                os.system('python -m spacy download en_core_web_lg')
            except Exception as ex:
                st.error(f'Error: {ex}')
            else:
                st.info('Accuracy Model Downloaded!')
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

# -------------------------------------------------------------------------------------------------------------------- #
# |                                           DATA LOADING AND PROCESSING                                            | #
# -------------------------------------------------------------------------------------------------------------------- #
        if st.button('Summarise Text', key='runner'):
            try:
                # CLEAN UP AND STANDARDISE DATAFRAMES
                DATA = DATA[[DATA_COLUMN]]
                DATA['SUMMARY'] = np.nan
                DATA = DATA.astype(str)
            except KeyError:
                st.error('Warning: CLEANED CONTENT is not found in the file uploaded. Try again.')

            if not DATA.empty:
                stopwords = list(STOP_WORDS)
                pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
                if NLP_MODEL == 'en_core_web_sm':
                    NLP = spacy.load('en_core_web_sm')
                elif NLP_MODEL == 'en_core_web_lg':
                    NLP = spacy.load('en_core_web_lg')

                try:
                    DATA['SUMMARY'] = DATA[DATA_COLUMN].apply(lambda x: summarise(x, stopwords, pos_tag, NLP, SENT_LEN))
                except Exception as e:
                    st.error(e)

# -------------------------------------------------------------------------------------------------------------------- #
# +                                                   DISPLAY DATA                                                   + #
# -------------------------------------------------------------------------------------------------------------------- #
            if VERBOSE:
                st.markdown('## Summarised Data')
                if VERBOSITY != 0:
                    try:
                        st.dataframe(DATA['SUMMARY'].head(VERBOSITY), height=400, width=800)
                    except RuntimeError:
                        st.warning(
                            'Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                        st.dataframe(DATA['SUMMARY'].head(10), height=400, width=800)
                    else:
                        if ADVANCED_ANALYSIS:
                            with st.expander('Advanced Profile Report'):
                                st_profile_report(DATA.profile_report(
                                    explorative=True,
                                    minimal=True
                                ))
                else:
                    try:
                        st.dataframe(DATA['SUMMARY'], height=400, width=800)
                    except RuntimeError:
                        st.warning(
                            'Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                        st.dataframe(DATA['SUMMARY'].head(10), height=400, width=800)
                    else:
                        if ADVANCED_ANALYSIS:
                            with st.expander('Advanced Profile Report'):
                                st_profile_report(DATA.profile_report(
                                    explorative=True,
                                    minimal=True
                                ))
            else:
                st.error('Warning: Data is processed wrongly. Try again.')
# -------------------------------------------------------------------------------------------------------------------- #
# +                                                     SAVE DATA                                                    + #
# -------------------------------------------------------------------------------------------------------------------- #
            if SAVE:
                st.markdown('## Download Summarised Data')
                st.markdown('Download summarised data from [downloads/summarised.csv]'
                            '(downloads/summarised.csv)')
                DATA.to_csv(str(DOWNLOAD_PATH / 'summarised.csv'))
            else:
                st.error('Warning: Data is processed wrongly. Try again.')


# -------------------------------------------------------------------------------------------------------------------- #
# |                                              SENTIMENT ANALYSIS                                                  | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif APP_MODE == 'Analyse Sentiment':
        st.markdown('# Sentiment Analysis')
# -------------------------------------------------------------------------------------------------------------------- #
# |                                               PROCESSING FLAGS                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
        st.markdown('## Flags')
        BACKEND_ANALYSER = st.selectbox('Choose the backend engine used to conduct sentiment analysis on your text',
                                        ('VADER', 'TextBlob'))
        SAVE = st.checkbox('Output to CSV file?')
        VERBOSE = st.checkbox('Print out the outputs to screen?')
        if VERBOSE:
            ADVANCED_ANALYSIS = st.checkbox('Display Advanced DataFrame Statistics?')
        elif VERBOSE and BACKEND_ANALYSER == 'VADER':
            VERBOSITY = st.slider('Data points', key='Data points to display?', min_value=1, max_value=1000, value=20)

# -------------------------------------------------------------------------------------------------------------------- #
# |                                           DATA LOADING AND PROCESSING                                            | #
# |                                                                                                                  | #
# |                             FIRST SECTION USES VADER (RECOMMENDED) AND THE SECOND                                | #
# |                                   SECTION USES TEXTBLOB (ALTERNATIVE OPTION)                                     | #
# -------------------------------------------------------------------------------------------------------------------- #
        if st.button('Start Analysis', key='analysis'):
            if BACKEND_ANALYSER == 'VADER':
                DATA['VADER SENTIMENT TEXT'] = DATA[DATA_COLUMN].str.lower().str.replace("'", ''). \
                    str.replace(r'[^\w\s]', ' ').str.replace(r" \d+", " ").str.replace(' +', ' ').str.strip()

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
                # LAMBDA FUNCTION
                polarity = lambda x: TextBlob(x).sentiment.polarity
                subjectivity = lambda x: TextBlob(x).sentiment.subjectivity

                # APPLY LAMBDAS ON THE DATAFRAME
                DATA['POLARITY'] = DATA[DATA_COLUMN].apply(polarity)
                DATA['SUBJECTIVITY'] = DATA[DATA_COLUMN].apply(subjectivity)

# -------------------------------------------------------------------------------------------------------------------- #
# +                                                  DISPLAY DATA                                                    + #
# -------------------------------------------------------------------------------------------------------------------- #
            if VERBOSE:
                if BACKEND_ANALYSER == 'VADER':
                    if 'VADER SENTIMENT' or 'VADER SCORE' in DATA.columns:
                        if VERBOSITY != 0:
                            try:
                                st.markdown('DataFrame')
                                st.dataframe(DATA.head(VERBOSITY), height=600, width=800)
                            except RuntimeError:
                                st.warning('Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                                st.dataframe(DATA.head(10), height=600, width=800)
                            else:
                                if ADVANCED_ANALYSIS:
                                    with st.expander('Advanced Profile Report'):
                                        st_profile_report(DATA.profile_report(
                                            explorative=True,
                                            minimal=True))
                        else:
                            try:
                                st.markdown('## DataFrame')
                                st.dataframe(DATA, height=600, width=800)
                            except RuntimeError:
                                st.warning('Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                                st.dataframe(DATA.head(10), height=600, width=800)
                            else:
                                if ADVANCED_ANALYSIS:
                                    with st.expander('Advanced Profile Report'):
                                        st_profile_report(DATA.profile_report(
                                            explorative=True,
                                            minimal=True))
                    else:
                        st.error('Warning: An error is made in the processing of the data. Try again.')
                elif BACKEND_ANALYSER == 'TextBlob':
                    if 'POLARITY' or 'SUBJECTIVITY' in DATA.columns:
                        st.markdown('## Visualise Polarity VS Subjectivity')
                        fig = px.scatter(x=DATA['SUBJECTIVITY'], y=DATA['POLARITY'])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error('Warning: An error is made in the processing of the data. Try again.')

# -------------------------------------------------------------------------------------------------------------------- #
# +                                                     SAVE DATA                                                    + #
# -------------------------------------------------------------------------------------------------------------------- #
            if SAVE:
                st.markdown('## Download Data')
                st.markdown('Download sentiment data from [downloads/sentiment_scores.csv]'
                            '(downloads/sentiment_scores.csv)')
                DATA.to_csv(str(DOWNLOAD_PATH / "sentiment_scores.csv"))


# -------------------------------------------------------------------------------------------------------------------- #
# |                                                TOPIC MODELLING                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif APP_MODE == 'Topic Modelling':
        st.markdown('# Topic Modelling')
# -------------------------------------------------------------------------------------------------------------------- #
# |                                                PROCESSING FLAGS                                                  | #
# -------------------------------------------------------------------------------------------------------------------- #
        st.markdown('## Flags')
        NUM_TOPICS = st.number_input('Define number of topics to extract from each piece of text',
                                     min_value=1,
                                     max_value=100,
                                     value=10)
        DOCUMENT_ID = st.number_input('Define the S/N of the document you wish to analyse',
                                      min_value=0,
                                      max_value=len(DATA[[DATA_COLUMN]]),
                                      value=0)
        SAVE = st.checkbox('Output to CSV file?')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                            DATA LOADING AND PROCESSING                                           | #
# -------------------------------------------------------------------------------------------------------------------- #
        if st.button('Start Modelling', key='topic'):
            if not DATA.empty:
                new_data = DATA[[DATA_COLUMN]]
                new_list = new_data[DATA_COLUMN].values.tolist()

                data_words = list(sent2word(new_list[DOCUMENT_ID]))
                data_words = stopwordRemover(data_words)

                id2word = corpora.Dictionary(data_words)
                texts = data_words
                corpus = [id2word.doc2bow(text) for text in texts]

                # BUILD MODEL
                LDA_MODEL = gensim.models.LdaMulticore(corpus=corpus,
                                                       id2word=id2word,
                                                       num_topics=NUM_TOPICS)
                TOPIC_LDA = pd.DataFrame(data=LDA_MODEL.get_topics())
                TOPIC_LDA_STR = LDA_MODEL.print_topics(num_topics=NUM_TOPICS)
                TOPIC_LDA_STR = list(TOPIC_LDA_STR)

                temp_dict = {}
                for data in TOPIC_LDA_STR:
                    temp_dict[data[0]] = data[1]

                print(temp_dict)
                TOPIC_LDA_STR = pd.DataFrame(data=temp_dict.values(), index=[i for i in range(len(TOPIC_LDA_STR))])
                document_lda = LDA_MODEL[corpus]
                LDA_DATA = pd.DataFrame(data=document_lda)

                # MAKE THE PLOTS
                LDA_VIS = pyLDAvis.gensim_models.prepare(LDA_MODEL,
                                                         corpus,
                                                         id2word)

                # FINALISE THE LISTS
                FINALISED_DATA_LIST = [
                    (TOPIC_LDA, 'Term Topic Matrix (Pickle)', 'term-topic-matrix.pkl', 'pkl'),
                    (TOPIC_LDA_STR, 'Term Topic Matrix Text', 'term-topic-matrix-text.csv', 'csv'),
                    (LDA_MODEL, 'LDA Model', 'LDA_model', 'model'),
                    (LDA_DATA, 'LDA Data (Pickle)', 'lda_data.pkl', 'pkl'),
                    (LDA_VIS, 'Model Visualisation', 'model_images.html', 'html')
                ]

# -------------------------------------------------------------------------------------------------------------------- #
# +                                                     SAVE DATA                                                    + #
# -------------------------------------------------------------------------------------------------------------------- #
                if SAVE:
                    st.markdown('## Download Data')
                    for data in FINALISED_DATA_LIST:
                        st.markdown(f'### {data[1]}')
                        st.markdown(f'Download data from [downloads/{data[2]}](downloads/{data[2]})')
                        if data[3] == 'pkl':
                            data[0].to_pickle(str(DOWNLOAD_PATH / data[2]))
                        elif data[3] == 'csv':
                            data[0].to_csv(str(DOWNLOAD_PATH / data[2]))
                        elif data[3] == 'model':
                            data[0].save(fname=str(DOWNLOAD_PATH / data[2]))
                        elif data[3] == 'html':
                            pyLDAvis.save_html(data[0], str(DOWNLOAD_PATH / data[2]))
