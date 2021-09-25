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

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from spacy.lang.en.stop_words import STOP_WORDS
from wordcloud import WordCloud
from textblob import TextBlob
from utils import csp_downloaders
from utils.helper import readFile, summarise, sent2word, stopwordRemover

# -------------------------------------------------------------------------------------------------------------------- #
# |                                               BASE DOWNLOAD PATH                                                 | #
# -------------------------------------------------------------------------------------------------------------------- #
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
DOWNLOAD_PATH = (STREAMLIT_STATIC_PATH / "downloads")
DATA = pd.DataFrame()


# -------------------------------------------------------------------------------------------------------------------- #
# |                                              MAIN APP FUNCTIONALITY                                              | #
# -------------------------------------------------------------------------------------------------------------------- #
def app():
    """
    Main function that will be called when the app is run
    """

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                             GLOBAL VARIABLES                                                 | #
    # ---------------------------------------------------------------------------------------------------------------- #
    global FILE, MODE, DATA_PATH, DATA, CSP, SAVE, VERBOSE, VERBOSITY, APP_MODE, BACKEND_ANALYSER, MAX_WORDS, \
        CONTOUR_WIDTH, DATA, MODEL_LEVEL, SENT_LEN, NUM_TOPICS, LDA_MODEL, MODEL, DOCUMENT_ID

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                                  INIT                                                        | #
    # ---------------------------------------------------------------------------------------------------------------- #
    # CREATE THE DOWNLOAD PATH WHERE FILES CREATED WILL BE STORED AND AVAILABLE FOR DOWNLOADING
    if not DOWNLOAD_PATH.is_dir():
        DOWNLOAD_PATH.mkdir()

    st.title('NLP Toolkit')
    st.markdown('## Init\n'
                'This module uses the gensim and spaCy package to conduct the necessary NLP preprocessing and '
                'analysis tasks for users to make sense of the text they pass into app. Note that this app requires '
                'the data to be decently cleaned; if you have not done so, run the Load, Clean adn Visualise module '
                'and save the cleaned and tokenized data onto your workstation. Those files may come in useful in '
                'the functionality of this app. Additionally, this module will only function if the column name '
                '"CLEANED CONTENT" is present. Kindly use the template on the left side bar to key in your '
                'information (select the "General" option).\n\n'
                '## Analysis Functions\n'
                'Select the NLP functions which you wish to apply to your dataset.')
    APP_MODE = st.selectbox('Select the NLP Operation to execute',
                            ('Word Cloud', 'Summarise', 'Analyse Sentiment', 'Topic Modelling'))
    st.markdown('## Upload Data\n'
                'Due to limitations imposed by the file uploader widget, only files smaller than 200 MB can be loaded '
                'with the widget. If your files are larger than 200 MB, please select "Large File(s)" and select the '
                'CSP you are using to host and store your data. Next, select the file format you wish to upload.')

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                                 TEMPLATE                                                     | #
    # ---------------------------------------------------------------------------------------------------------------- #
    FILE = st.selectbox('Select the type of file to load', ('Small File(s)', 'Large File(s)'))
    MODE = st.selectbox('Define the data input format', ('CSV', 'XLSX'))

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                               FILE UPLOADING                                                 | #
    # ---------------------------------------------------------------------------------------------------------------- #
    if FILE == 'Small File(s)':
        st.markdown('### Upload the file that you wish to analyse:\n')
        DATA_PATH = st.file_uploader('Load up a CSV/XLSX File containing the cleaned data', type=[MODE])
        if st.button('Load Data', key='Loader'):
            DATA = readFile(DATA_PATH, MODE)
            if not DATA.empty:
                st.success('Data Loaded!')

    elif FILE == 'Large File(s)':
        CSP = st.selectbox('CSP', ('None', 'Azure', 'Amazon', 'Google'))

        if CSP == 'Azure':
            azure = csp_downloaders.AzureDownloader()
            if st.button('Continue', key='az'):
                azure.downloadBlob()
                DATA = readFile(csp_downloaders.AZURE_DOWNLOAD_ABS_PATH, MODE)
                st.info('File Read!')

        elif CSP == 'Amazon':
            aws = csp_downloaders.AWSDownloader()
            if st.button('Continue', key='aws'):
                aws.downloadFile()
                DATA = readFile(csp_downloaders.AWS_FILE_NAME, MODE)
                st.info('File Read!')

        elif CSP == 'Google':
            gcs = csp_downloaders.GoogleDownloader()
            if st.button('Continue', key='gcs'):
                gcs.downloadBlob()
                DATA = readFile(csp_downloaders.GOOGLE_DESTINATION_FILE_NAME, MODE)
                st.info('File Read!')


# -------------------------------------------------------------------------------------------------------------------- #
# |                                            WORD CLOUD VISUALISATION                                              | #
# -------------------------------------------------------------------------------------------------------------------- #
    if APP_MODE == 'Word Cloud':
        # ------------------------------------------------------------------------------------------------------------ #
        # |                                           PROCESSING FLAGS                                               | #
        # ------------------------------------------------------------------------------------------------------------ #
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

        # ------------------------------------------------------------------------------------------------------------ #
        # |                                       DATA LOADING AND PROCESSING                                        | #
        # ------------------------------------------------------------------------------------------------------------ #
        if st.button('Generate Word Cloud', key='wc'):
            DATA = DATA[['CLEANED CONTENT']]
            temp_list = DATA['CLEANED CONTENT'].tolist()

            temp_text = ' '
            for i in range(len(temp_list)):
                temp_text = str(temp_list[i]) + temp_text

            wc = WordCloud(background_color='white',
                           max_words=MAX_WORDS,
                           contour_width=CONTOUR_WIDTH,
                           contour_color='steelblue')
            wc.generate(temp_text)

            st.markdown('## Wordcloud')
            st.image(wc.to_image(), width=None)

            if SAVE:
                st.markdown('## Download Image')
                st.markdown('Download image from [downloads/wordcloud.png](downloads/wordcloud.png)')
                wc.to_file(str(DOWNLOAD_PATH / 'wordcloud.png'))


# -------------------------------------------------------------------------------------------------------------------- #
# |                                                 SUMMARIZATION                                                    | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif APP_MODE == 'Summarise':
        # ------------------------------------------------------------------------------------------------------------ #
        # |                                           PROCESSING FLAGS                                               | #
        # ------------------------------------------------------------------------------------------------------------ #
        st.markdown('## Flags')
        SENT_LEN = st.number_input('Enter the total number of sentences to summarise text to',
                                   min_value=1,
                                   max_value=100,
                                   value=3)
        SAVE = st.checkbox('Output to CSV file?')
        VERBOSE = st.checkbox('Print out DataFrames?')
        if VERBOSE:
            VERBOSITY = st.slider('Data points', key='Data points to display?', min_value=1, max_value=1000, value=20)

        # ------------------------------------------------------------------------------------------------------------ #
        # |                                       DATA LOADING AND PROCESSING                                        | #
        # ------------------------------------------------------------------------------------------------------------ #
        if st.button('Summarise Text', key='runner'):
            try:
                # CLEAN UP AND STANDARDISE DATAFRAMES
                DATA = DATA[['CLEANED CONTENT']]
                DATA['SUMMARY'] = np.nan
                DATA = DATA.astype(str)
            except KeyError:
                st.error('Warning: CLEANED CONTENT is not found in the file uploaded. Try again.')

            if not DATA.empty:
                stopwords = list(STOP_WORDS)
                pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
                nlp = spacy.load('en_core_web_sm')

                try:
                    DATA['SUMMARY'] = DATA['CLEANED CONTENT'].apply(lambda x: summarise(x, stopwords,
                                                                                        pos_tag, nlp,
                                                                                        SENT_LEN))
                except Exception as e:
                    st.error(e)

            # -------------------------------------------------------------------------------------------------------- #
            # +                                             DISPLAY DATA                                             + #
            # -------------------------------------------------------------------------------------------------------- #
            if VERBOSE:
                st.markdown('## Summarised Data')
                if VERBOSITY != 0:
                    try:
                        st.dataframe(DATA['SUMMARY'].head(VERBOSITY))
                    except RuntimeError:
                        st.warning(
                            'Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                        st.dataframe(DATA['SUMMARY'].head(10))
                else:
                    try:
                        st.dataframe(DATA['SUMMARY'])
                    except RuntimeError:
                        st.warning(
                            'Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                        st.dataframe(DATA['SUMMARY'].head(10))
            else:
                st.error('Warning: Data is processed wrongly. Try again.')
            # -------------------------------------------------------------------------------------------------------- #
            # +                                               SAVE DATA                                              + #
            # -------------------------------------------------------------------------------------------------------- #
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
        # ------------------------------------------------------------------------------------------------------------ #
        # |                                           PROCESSING FLAGS                                               | #
        # ------------------------------------------------------------------------------------------------------------ #
        st.markdown('## Flags')
        BACKEND_ANALYSER = st.selectbox('Choose the backend engine used to conduct sentiment analysis on your text',
                                        ('VADER', 'TextBlob'))
        SAVE = st.checkbox('Output to CSV file?')
        VERBOSE = st.checkbox('Print out the outputs to screen?')
        if VERBOSE and BACKEND_ANALYSER == 'VADER':
            VERBOSITY = st.slider('Data points', key='Data points to display?', min_value=1, max_value=1000, value=20)

        # ------------------------------------------------------------------------------------------------------------ #
        # |                                       DATA LOADING AND PROCESSING                                        | #
        # ------------------------------------------------------------------------------------------------------------ #
        if st.button('Start Analysis', key='analysis'):
            # -------------------------------------------------------------------------------------------------------- #
            # +                                                 VADER                                                + #
            # -------------------------------------------------------------------------------------------------------- #
            if BACKEND_ANALYSER == 'VADER':
                DATA['VADER SENTIMENT TEXT'] = DATA['CLEANED CONTENT'].str.lower().str.replace("'", ''). \
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

            # -------------------------------------------------------------------------------------------------------- #
            # +                                               TEXTBLOB                                               + #
            # -------------------------------------------------------------------------------------------------------- #
            elif BACKEND_ANALYSER == 'TextBlob':
                # LAMBDA FUNCTION
                polarity = lambda x: TextBlob(x).sentiment.polarity
                subjectivity = lambda x: TextBlob(x).sentiment.subjectivity

                # APPLY LAMBDAS ON THE DATAFRAME
                DATA['POLARITY'] = DATA['CLEANED CONTENT'].apply(polarity)
                DATA['SUBJECTIVITY'] = DATA['CLEANED CONTENT'].apply(subjectivity)

            # -------------------------------------------------------------------------------------------------------- #
            # +                                             DISPLAY DATA                                             + #
            # -------------------------------------------------------------------------------------------------------- #
            if VERBOSE:
                if BACKEND_ANALYSER == 'VADER':
                    if 'VADER SENTIMENT' or 'VADER SCORE' in DATA.columns:
                        if VERBOSITY != 0:
                            try:
                                st.markdown('DataFrame')
                                st.dataframe(DATA.head(VERBOSITY))
                            except RuntimeError:
                                st.warning('Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                                st.dataframe(DATA.head(10))
                        else:
                            try:
                                st.markdown('## DataFrame')
                                st.dataframe(DATA)
                            except RuntimeError:
                                st.warning('Warning: Size of DataFrame is too large. Defaulting to 10 data points...')
                                st.dataframe(DATA.head(10))
                    else:
                        st.error('Warning: An error is made in the processing of the data. Try again.')
                elif BACKEND_ANALYSER == 'TextBlob':
                    if 'POLARITY' or 'SUBJECTIVITY':
                        st.markdown('## Visualise Polarity VS Subjectivity')
                        fig = px.scatter(x=DATA['SUBJECTIVITY'], y=DATA['POLARITY'])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error('Warning: An error is made in the processing of the data. Try again.')

            # -------------------------------------------------------------------------------------------------------- #
            # +                                               SAVE DATA                                              + #
            # -------------------------------------------------------------------------------------------------------- #
            if SAVE:
                st.markdown('## Download Data')
                st.markdown('Download sentiment data from [downloads/sentiment_scores.csv]'
                            '(downloads/sentiment_scores.csv)')
                DATA.to_csv(str(DOWNLOAD_PATH / "sentiment_scores.csv"))


# -------------------------------------------------------------------------------------------------------------------- #
# |                                                TOPIC MODELLING                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif APP_MODE == 'Topic Modelling':
        # ------------------------------------------------------------------------------------------------------------ #
        # |                                           PROCESSING FLAGS                                               | #
        # ------------------------------------------------------------------------------------------------------------ #
        st.markdown('## Flags')
        NUM_TOPICS = st.number_input('Define number of topics to extract from each piece of text',
                                     min_value=1,
                                     max_value=100,
                                     value=10)
        DOCUMENT_ID = st.number_input('Define the S/N of the document you wish to analyse',
                                      min_value=0,
                                      max_value=len(DATA[['CLEANED CONTENT']]),
                                      value=0)
        SAVE = st.checkbox('Output to CSV file?')

        # ------------------------------------------------------------------------------------------------------------ #
        # |                                       DATA LOADING AND PROCESSING                                        | #
        # ------------------------------------------------------------------------------------------------------------ #
        if st.button('Start Modelling', key='topic'):
            if not DATA.empty:
                new_data = DATA[['CLEANED CONTENT']]
                new_list = new_data['CLEANED CONTENT'].values.tolist()

                data_words = list(sent2word(new_list[DOCUMENT_ID]))
                data_words = stopwordRemover(data_words)

                id2word = corpora.Dictionary(data_words)
                texts = data_words
                corpus = [id2word.doc2bow(text) for text in texts]

                # BUILD MODEL
                LDA_MODEL = gensim.models.LdaMulticore(corpus=corpus,
                                                       id2word=id2word,
                                                       num_topics=NUM_TOPICS)
                topic_lda = pd.DataFrame(data=LDA_MODEL.get_topics())
                topic_lda_str = LDA_MODEL.print_topics(num_topics=NUM_TOPICS)
                topic_lda_str = list(topic_lda_str)

                temp_dict = {}
                for data in topic_lda_str:
                    temp_dict[data[0]] = data[1]

                print(temp_dict)
                topic_lda_str = pd.DataFrame(data=temp_dict.values(), index=[i for i in range(len(topic_lda_str))])
                document_lda = LDA_MODEL[corpus]
                lda_data = pd.DataFrame(data=document_lda)

                # MAKE THE PLOTS
                lda_vis = pyLDAvis.gensim_models.prepare(LDA_MODEL,
                                                         corpus,
                                                         id2word)

                # ---------------------------------------------------------------------------------------------------- #
                # +                                             SAVE DATA                                            + #
                # ---------------------------------------------------------------------------------------------------- #
                if SAVE:
                    st.markdown('## Download Data')
                    st.markdown('Download Term-Topic Matrix from [downloads/term-topic-matrix.pkl]'
                                '(downloads/term-topic-matrix.pkl) and [downloads/term-topic-matrix-text.csv]'
                                '(downloads/term-topic-matrix-text.csv)')
                    st.markdown('Download LDA Model from [downloads/LDA_model](downloads/LDA_model)')
                    st.markdown('Download LDA Model Data from [downloads/lda_data.csv](downloads/lda_data.csv)')

                    topic_lda.to_pickle(str(DOWNLOAD_PATH / 'term-topic-matrix.pkl'))
                    topic_lda_str.to_csv(str(DOWNLOAD_PATH / 'term-topic-matrix-text.csv'))
                    LDA_MODEL.save(fname=str(DOWNLOAD_PATH / 'LDA_model'))
                    lda_data.to_pickle(str(DOWNLOAD_PATH / 'lda_data.pkl'))
                    st.markdown('Download HTML file from [downloads/model_images.html](downloads/model_images.html)')
                    pyLDAvis.save_html(lda_vis, str(DOWNLOAD_PATH / 'model_images.html'))
