"""
This module allows the user to conduct basic NLP analysis functions on the preprocessed data using the spaCy module
This module uses CPU-optimised pipelines and hence a GPU is optional in this module
"""

# -------------------------------------------------------------------------------------------------------------------- #
# |                                         IMPORT RELEVANT LIBRARIES                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
import multiprocessing
import os
import numpy as np
import pandas as pd
import spacy
import streamlit as st
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
import nltk
import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.sklearn
import streamlit.components.v1
import torch
import matplotlib.pyplot as plt
import transformers

from streamlit_tags import st_tags
from config import toolkit
from operator import itemgetter
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, AutoModelForSequenceClassification
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
from utils.helper import readFile, summarise, modelIterator, printDataFrame, dominantTopic, prettyDownload


# -------------------------------------------------------------------------------------------------------------------- #
# |                                             MAIN APP FUNCTIONALITY                                               | #
# -------------------------------------------------------------------------------------------------------------------- #
def app():
    """
    Main function that will be called when the app is run
    """

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                    INIT                                                          | #
# -------------------------------------------------------------------------------------------------------------------- #
    st.title('NLP Toolkit')
    st.markdown('## Init\n'
                'This module uses the *spaCy* package to conduct the necessary NLP preprocessing and '
                'analysis tasks for users to make sense of the text they pass into app. Note that this app requires '
                'the data to be decently cleaned; if you have not done so, run the *Load, Clean and Visualise* module '
                'and save the cleaned  data onto your workstation. Those files may come in useful in '
                'the functionality of this app.\n\n')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                              FUNCTION SELECTOR                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    st.markdown('## NLP Operations\n'
                'Select the NLP functions which you wish to execute.')
    toolkit['APP_MODE'] = st.selectbox('Select the NLP Operation to execute',
                                       ('Topic Modelling', 'Topic Classification', 'Analyse Sentiment', 'Word Cloud',
                                        'Named Entity Recognition', 'POS Tagging', 'Summarise'))
    st.info(f'**{toolkit["APP_MODE"]}** Selected')

    if toolkit['APP_MODE'] != 'News Classifier':
        st.markdown('## Upload Data\n')
        col1, col1_ = st.columns(2)
        toolkit['FILE'] = col1.selectbox('Origin of Data File', ('Local', 'Online'),
                                         help='Choose "Local" if you wish to upload a file from your machine or choose '
                                              '"Online" if you wish to pull a file from any one of the supported Cloud '
                                              'Service Providers.')
        toolkit['MODE'] = col1_.selectbox('Define the Data Input Format', ('CSV', 'XLSX', 'PKL', 'JSON', 'HDF5'))

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                 FILE UPLOADING                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
        if toolkit['FILE'] == 'Local':
            toolkit['DATA_PATH'] = st.file_uploader(f'Load {toolkit["MODE"]} File', type=[toolkit['MODE']])
            if toolkit['DATA_PATH'] is not None:
                toolkit['DATA'] = readFile(toolkit['DATA_PATH'], toolkit['MODE'])
                if not toolkit['DATA'].empty:
                    toolkit['DATA'] = toolkit['DATA'].astype(str)
                    toolkit['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored',
                                                          list(toolkit['DATA'].columns))
                    st.success(f'Data Loaded from {toolkit["DATA_COLUMN"]}!')
            else:
                toolkit['DATA'] = pd.DataFrame()

        elif toolkit['FILE'] == 'Online':
            st.info(f'File Format Selected: **{toolkit["MODE"]}**')
            toolkit['CSP'] = st.selectbox('CSP', ('Select a CSP', 'Azure', 'Amazon', 'Google'))

            if toolkit['CSP'] == 'Azure':
                azure = csp_downloaders.AzureDownloader()
                if azure.SUCCESSFUL:
                    try:
                        azure.downloadBlob()
                        toolkit['DATA'] = readFile(azure.AZURE_DOWNLOAD_PATH, toolkit['MODE'])
                    except Exception as ex:
                        st.error(f'Error: {ex}. Try again.')

                if not toolkit['DATA'].empty:
                    toolkit['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored',
                                                          list(toolkit['DATA'].columns))
                    st.success(f'Data Loaded from {toolkit["DATA_COLUMN"]}!')

            elif toolkit['CSP'] == 'Amazon':
                aws = csp_downloaders.AWSDownloader()
                if aws.SUCCESSFUL:
                    try:
                        aws.downloadFile()
                        toolkit['DATA'] = readFile(aws.AWS_FILE_NAME, toolkit['MODE'])
                    except Exception as ex:
                        st.error(f'Error: {ex}. Try again.')

                if not toolkit['DATA'].empty:
                    toolkit['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored',
                                                          list(toolkit['DATA'].columns))
                    st.success(f'Data Loaded from {toolkit["DATA_COLUMN"]}!')

            elif toolkit['CSP'] == 'Google':
                gcs = csp_downloaders.GoogleDownloader()
                if gcs.SUCCESSFUL:
                    try:
                        gcs.downloadBlob()
                        toolkit['DATA'] = readFile(gcs.GOOGLE_DESTINATION_FILE_NAME, toolkit['MODE'])
                    except Exception as ex:
                        st.error(f'Error: {ex}. Try again.')

                if not toolkit['DATA'].empty:
                    toolkit['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored',
                                                          list(toolkit['DATA'].columns))
                    st.success(f'Data Loaded from {toolkit["DATA_COLUMN"]}!')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                            WORD CLOUD VISUALISATION                                              | #
# -------------------------------------------------------------------------------------------------------------------- #
    if toolkit['APP_MODE'] == 'Word Cloud':
        st.markdown('---')
        st.markdown('# Word Cloud Generation\n'
                    'This module takes in a long list of documents and converts it into a WordCloud representation '
                    'of all the documents.\n\n'
                    'Note that the documents should not be tokenized, but it should be cleaned and lemmatized to '
                    'avoid double-counting words.')

        # FLAGS
        st.markdown('## Options')
        toolkit['SAVE'] = st.checkbox('Save Outputs?',
                                      help='Due to the possibility of files with the same file name and content being '
                                           'downloaded again, a unique file identifier is tacked onto the filename.')

        col1, col1_ = st.columns(2)
        toolkit['MAX_WORDS'] = col1.number_input('Key in the maximum number of words to display',
                                                 min_value=2,
                                                 max_value=1000,
                                                 value=200)
        toolkit['CONTOUR_WIDTH'] = col1_.number_input('Key in the contour width of your WordCloud',
                                                      min_value=1,
                                                      max_value=10,
                                                      value=3)
        toolkit['WIDTH'] = col1.number_input('Key in the Width of the WordCloud image generated',
                                             min_value=1,
                                             max_value=100000,
                                             value=800)
        toolkit['HEIGHT'] = col1_.number_input('Key in the Height of the WordCloud image generated',
                                               min_value=1,
                                               max_value=100000,
                                               value=400)

        # MAIN DATA PROCESSING
        if st.button('Generate Word Cloud', key='wc'):
            if not toolkit['DATA'].empty:
                toolkit['DATA'] = toolkit['DATA'][[toolkit['DATA_COLUMN']]]
                wc = WordCloud(background_color='white',
                               max_words=toolkit['MAX_WORDS'],
                               contour_width=toolkit['CONTOUR_WIDTH'],
                               width=toolkit['WIDTH'],
                               height=toolkit['HEIGHT'],
                               contour_color='steelblue')
                wc.generate(' '.join(toolkit['DATA'][toolkit['DATA_COLUMN']]))

                st.markdown('## Wordcloud For Text Inputted')
                st.image(wc.to_image(), width=None)

                if toolkit['SAVE']:
                    st.markdown('---')
                    st.markdown('## Download Image')
                    st.markdown(prettyDownload(
                        object_to_download=wc,
                        download_filename='wordcloud.png',
                        button_text='Download Queried Data',
                        override_index=False),
                        unsafe_allow_html=True
                    )
            else:
                st.error('Error: Data not loaded properly. Try again.')


# -------------------------------------------------------------------------------------------------------------------- #
# |                                            NAMED ENTITY RECOGNITION                                              | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif toolkit['APP_MODE'] == 'Named Entity Recognition':
        st.markdown('---')
        st.markdown('# Named Entity Recognition')
        st.markdown('Note that this module takes a long time to process a long piece of text. If you intend to process '
                    'large chunks of text, prepare to wait for hours for the NER Tagging process to finish. We are '
                    'looking to implement multiprocessing into the app to optimise it, but so far multiprocessing '
                    'does not seem to be fully supported in Streamlit.\n\n'
                    'In the meantime, it may be better to process your data in smaller batches to speed up your '
                    'workflow.')

        # FLAGS
        st.markdown('## Options')
        st.markdown('Due to limits imposed on the visualisation engine and to avoid cluttering of the page with '
                    'outputs, you will only be able to visualise the NER outputs for a single piece of text at any '
                    'one point. However, you will still be able to download a text/html file containing '
                    'the outputs for you to save onto your disks.\n'
                    '### NLP Models\n'
                    'Select one model to use for your NLP Processing. Choose en_core_web_sm for a model that is '
                    'optimised for efficiency or en_core_web_lg for a model that is optimised for accuracy.')
        toolkit['NLP_MODEL'] = st.radio('Select spaCy model', ('en_core_web_sm', 'en_core_web_lg'))
        if toolkit['NLP_MODEL'] == 'en_core_web_sm':
            try:
                toolkit['NLP'] = spacy.load('en_core_web_sm')
            except OSError:
                st.warning('Model not found, downloading...')
                try:
                    os.system('python -m spacy download en_core_web_sm')
                except Exception as ex:
                    st.error(f'Unable to download Model. Error: {ex}')
            except Exception as ex:
                st.error(f'Unknown Error: {ex}. Try again.')
            else:
                st.info('**Efficiency Model** Loaded!')
        elif toolkit['NLP_MODEL'] == 'en_core_web_lg':
            try:
                toolkit['NLP'] = spacy.load('en_core_web_lg')
            except OSError:
                st.warning('Model not found, downloading...')
                try:
                    os.system('python -m spacy download en_core_web_lg')
                except Exception as ex:
                    st.error(f'Unable to download Model. Error: {ex}')
            except Exception as ex:
                st.error(f'Unknown Error: {ex}. Try again.')
            else:
                st.info('**Accuracy Model** Loaded!')
        toolkit['VERBOSE'] = st.checkbox('Display Outputs?')
        if toolkit['VERBOSE']:
            toolkit['VERBOSITY'] = st.slider('Choose Number of Data Points to Display',
                                             min_value=0,
                                             max_value=1000,
                                             value=20,
                                             help='Select 0 to display all Data Points')
            toolkit['ONE_DATAPOINT'] = st.checkbox('Visualise One Data Point?')
            if toolkit['ONE_DATAPOINT']:
                toolkit['DATAPOINT_SELECTOR'] = st.selectbox('Choose Data Point From Data', range(len(toolkit['DATA'])))
            else:
                st.info('You are conducting NER on the entire dataset. Only DataFrame is printed. NER output will be '
                        'automatically saved.')
            toolkit['ADVANCED_ANALYSIS'] = st.checkbox('Display Advanced DataFrame Statistics?',
                                                       help='This option will analyse your DataFrame and display '
                                                            'advanced statistics on it. Note that this will require '
                                                            'some time and processing power to complete. Deselect this '
                                                            'option if this if you do not require it.')
        toolkit['SAVE'] = st.checkbox('Save Outputs?', help='Due to the possibility of files with the same file name '
                                                            'and content being downloaded again, a unique file '
                                                            'identifier is tacked onto the filename.')
        if toolkit['SAVE']:
            if st.checkbox('Override Output Format?'):
                toolkit['OVERRIDE_FORMAT'] = st.selectbox('Overridden Output Format',
                                                          ('CSV', 'XLSX', 'PKL', 'JSON', 'HDF5'))
                if toolkit['OVERRIDE_FORMAT'] == toolkit['MODE']:
                    st.warning('Warning: Overridden Format is the same as Input Format')
            else:
                toolkit['OVERRIDE_FORMAT'] = None

        # MAIN PROCESSING
        if st.button('Conduct Named Entity Recognition', key='ner'):
            if not toolkit['DATA'].empty:
                # EFFICIENT NLP PIPING
                ner = []
                lab = []
                toolkit['DATA'] = toolkit['DATA'][[toolkit['DATA_COLUMN']]].astype(str)
                for doc in toolkit['NLP'].pipe(toolkit['DATA'][toolkit['DATA_COLUMN']].to_list(),
                                               disable=['tagger', 'parser', 'entity_linker', 'entity_ruler',
                                                        'textcat', 'textcat_multilabel', 'lemmatizer',
                                                        'morphologizer', 'attribute_ruler', 'senter',
                                                        'sentencizer', 'tok2vec', 'transformer'],
                                               batch_size=2000,
                                               n_process=1):
                    ner.append(str(list(zip([word.text for word in doc.ents], [word.label_ for word in doc.ents]))))
                    lab.append(str(list(set([word.label_ for word in doc.ents]))))
                toolkit['DATA']['NER'] = ner
                toolkit['DATA']['COMPILED_LABELS'] = lab

                if toolkit['VERBOSE']:
                    st.markdown('## NER DataFrame')
                    printDataFrame(data=toolkit['DATA'], verbose_level=toolkit['VERBOSITY'],
                                   advanced=toolkit['ADVANCED_ANALYSIS'])

                    if toolkit['ONE_DATAPOINT']:
                        verbose_data_copy = toolkit['DATA'].copy()
                        temp_df = verbose_data_copy[toolkit['DATA_COLUMN']][toolkit['DATAPOINT_SELECTOR']]
                        st.markdown('## DisplaCy Rendering')
                        st.info('If rendering is not clean, choose to save files generated and download the rendering '
                                'in HTML format.')
                        toolkit['SVG'] = displacy.render(list(toolkit['NLP'](str(temp_df)).sents),
                                                         style='ent',
                                                         page=True)
                        st.markdown(toolkit['SVG'], unsafe_allow_html=True)

                if toolkit['SAVE']:
                    st.markdown('---')
                    st.markdown('## Download Data')
                    if toolkit['OVERRIDE_FORMAT'] is not None:
                        st.markdown(
                            prettyDownload(
                                object_to_download=toolkit['DATA'],
                                download_filename=f'ner.{toolkit["OVERRIDE_FORMAT"].lower()}',
                                button_text=f'Download NER Data',
                                override_index=False,
                                format_=toolkit['OVERRIDE_FORMAT']),
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            prettyDownload(
                                object_to_download=toolkit['DATA'],
                                download_filename=f'ner.{toolkit["MODE"].lower()}',
                                button_text=f'Download NER Data',
                                override_index=False,
                                format_=toolkit['MODE']),
                            unsafe_allow_html=True
                        )
                    if toolkit['ONE_DATAPOINT']:
                        st.markdown(
                            prettyDownload(
                                object_to_download=toolkit['SVG'],
                                download_filename='rendering.html',
                                button_text=f'Download Rendering Data',
                                override_index=False),
                            unsafe_allow_html=True
                        )
            else:
                st.error('Error: Data not loaded properly. Try again.')


# -------------------------------------------------------------------------------------------------------------------- #
# |                                             PART OF SPEECH TAGGING                                               | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif toolkit['APP_MODE'] == 'POS Tagging':
        st.markdown('---')
        st.markdown('# POS Tagging')
        st.markdown('Note that this module takes a long time to process a long piece of text. If you intend to process '
                    'large chunks of text, prepare to wait for hours for the POS tagging process to finish. We are '
                    'looking to implement multiprocessing into the app to optimise it.\n\n'
                    'In the meantime, it may be better to process your data in smaller batches to speed up your '
                    'workflow.')

        # FLAGS
        st.markdown('## Options')
        st.markdown('Due to limits imposed on the visualisation engine and to avoid cluttering of the page with '
                    'outputs, you will only be able to visualise the NER outputs for a single piece of text at any '
                    'one point. However, you will still be able to download a text/html file containing '
                    'the outputs for you to save onto your disks.\n'
                    '### NLP Models\n'
                    'Select one model to use for your NLP Processing. Choose en_core_web_sm for a model that is '
                    'optimised for efficiency or en_core_web_lg for a model that is optimised for accuracy.')
        toolkit['NLP_MODEL'] = st.radio('Select spaCy model', ('en_core_web_sm', 'en_core_web_lg'))
        if toolkit['NLP_MODEL'] == 'en_core_web_sm':
            try:
                toolkit['NLP'] = spacy.load('en_core_web_sm')
            except OSError:
                st.warning('Model not found, downloading...')
                try:
                    os.system('python -m spacy download en_core_web_sm')
                except Exception as ex:
                    st.error(f'Unable to download Model. Error: {ex}')
            except Exception as ex:
                st.error(f'Unknown Error: {ex}. Try again.')
            else:
                st.info('**Efficiency Model** Loaded!')
        elif toolkit['NLP_MODEL'] == 'en_core_web_lg':
            try:
                toolkit['NLP'] = spacy.load('en_core_web_lg')
            except OSError:
                st.warning('Model not found, downloading...')
                try:
                    os.system('python -m spacy download en_core_web_lg')
                except Exception as ex:
                    st.error(f'Unable to download Model. Error: {ex}')
            except Exception as ex:
                st.error(f'Unknown Error: {ex}. Try again.')
            else:
                st.info('**Accuracy Model** Loaded!')
        toolkit['VERBOSE'] = st.checkbox('Display Outputs?')
        if toolkit['VERBOSE']:
            toolkit['VERBOSITY'] = st.slider('Choose Number of Data Points to Display',
                                             min_value=0,
                                             max_value=1000,
                                             value=20,
                                             help='Select 0 to display all Data Points')
            toolkit['ONE_DATAPOINT'] = st.checkbox('Visualise One Data Point?')
            toolkit['ADVANCED_ANALYSIS'] = st.checkbox('Display Advanced DataFrame Statistics?',
                                                       help='This option will analyse your DataFrame and display '
                                                            'advanced statistics on it. Note that this will require '
                                                            'some time and processing power to complete. Deselect this '
                                                            'option if this if you do not require it.')
            if toolkit['ONE_DATAPOINT']:
                toolkit['DATAPOINT_SELECTOR'] = st.selectbox('Choose Data Point From Data', range(len(toolkit['DATA'])))
                toolkit['COLOUR_BCKGD'] = st.color_picker('Choose Colour of Render Background', value='#000000')
                toolkit['COLOUR_TXT'] = st.color_picker('Choose Colour of Render Text', value='#ffffff')
            else:
                st.info('You are conducting POS on the entire dataset. Only DataFrame is printed. POS output will be '
                        'automatically saved.')
        toolkit['SAVE'] = st.checkbox('Save Outputs?',
                                      help='Due to the possibility of files with the same file name and content being '
                                           'downloaded again, a unique file identifier is tacked onto the filename.')
        if toolkit['SAVE']:
            if st.checkbox('Override Output Format?'):
                toolkit['OVERRIDE_FORMAT'] = st.selectbox('Overridden Output Format',
                                                          ('CSV', 'XLSX', 'PKL', 'JSON', 'HDF5'))
                if toolkit['OVERRIDE_FORMAT'] == toolkit['MODE']:
                    st.warning('Warning: Overridden Format is the same as Input Format')
            else:
                toolkit['OVERRIDE_FORMAT'] = None

        # MAIN PROCESSING
        if st.button('Start POS Tagging', key='pos'):
            # RESET OUTPUTS
            toolkit['SVG'] = None

            if not toolkit['DATA'].empty:
                # EFFICIENT NLP PIPING
                pos = []
                lab = []
                toolkit['DATA'] = toolkit['DATA'][[toolkit['DATA_COLUMN']]].astype(str)
                for doc in toolkit['NLP'].pipe(toolkit['DATA'][toolkit['DATA_COLUMN']].to_list(),
                                               disable=['ner', 'parser', 'entity_linker', 'entity_ruler',
                                                        'textcat', 'textcat_multilabel', 'lemmatizer',
                                                        'morphologizer', 'attribute_ruler', 'senter',
                                                        'sentencizer', 'tok2vec', 'transformer'],
                                               batch_size=2000,
                                               n_process=1):
                    pos.append(str(list(zip([str(word) for word in doc], [word.pos_ for word in doc]))))
                    lab.append(str(list(set([word.pos_ for word in doc]))))
                toolkit['DATA']['POS'] = pos
                toolkit['DATA']['COMPILED_LABELS'] = lab

                if toolkit['VERBOSE']:
                    st.markdown('## POS DataFrame')
                    printDataFrame(data=toolkit['DATA'], verbose_level=toolkit['VERBOSITY'],
                                   advanced=toolkit['ADVANCED_ANALYSIS'])

                    if toolkit['ONE_DATAPOINT']:
                        verbose_data_copy = toolkit['DATA'].copy()
                        temp_df = verbose_data_copy[toolkit['DATA_COLUMN']][toolkit['DATAPOINT_SELECTOR']]
                        st.info('Renders are not shown due to the sheer size of the image. Kindly save the render in '
                                'HTML format below to view it.')
                        toolkit['SVG'] = displacy.render(list(toolkit['NLP'](str(temp_df)).sents),
                                                         style='dep',
                                                         options={
                                                             'compact': True,
                                                             'color': toolkit['COLOUR_TXT'],
                                                             'bg': toolkit['COLOUR_BCKGD'],
                                                         })

                if toolkit['SAVE']:
                    st.markdown('---')
                    st.markdown('## Download Data')
                    if toolkit['OVERRIDE_FORMAT'] is not None:
                        st.markdown(prettyDownload(object_to_download=toolkit['DATA'],
                                                   download_filename=f'pos.{toolkit["OVERRIDE_FORMAT"].lower()}',
                                                   button_text=f'Download POS Data',
                                                   override_index=False,
                                                   format_=toolkit['OVERRIDE_FORMAT']),
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(prettyDownload(object_to_download=toolkit['DATA'],
                                                   download_filename=f'pos.{toolkit["MODE"].lower()}',
                                                   button_text=f'Download POS Data',
                                                   override_index=False,
                                                   format_=toolkit['MODE']),
                                    unsafe_allow_html=True)

                    if toolkit['VERBOSE'] and toolkit['ONE_DATAPOINT']:
                        st.markdown(prettyDownload(
                            object_to_download=toolkit['SVG'],
                            download_filename='rendering.html',
                            button_text=f'Download Rendering Data',
                            override_index=False),
                            unsafe_allow_html=True
                        )
            else:
                st.error('Error: Data not loaded properly. Try again.')


# -------------------------------------------------------------------------------------------------------------------- #
# |                                                 SUMMARIZATION                                                    | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif toolkit['APP_MODE'] == 'Summarise':
        st.markdown('---')
        st.markdown('# Summarization of Text')
        st.markdown('For this function, you are able to upload a piece of document or multiple pieces of documents '
                    'in a CSV file to create a summary for the documents of interest.\n\n'
                    'However, do note that this module takes a long time to process a long piece of text, '
                    'hence, it may be better to process your data in smaller batches to speed up your '
                    'workflow.\n\n'
                    'In an effort to enhance this module to provide users with meaningful summaries of their document, '
                    ' we have implemented two modes of summarization in this function, namely Basic and Advanced '
                    'Mode.\n')

        st.markdown('## Summary Complexity')
        st.markdown('**Basic Mode** uses the spaCy package to distill your documents into the specified number of '
                    'sentences. No machine learning model was used to produce a unique summary of the text.\n\n'
                    '**Advanced Mode** uses the Pytorch and Huggingface Transformers library to produce summaries '
                    'using Google\'s T5 Model.')
        toolkit['SUM_MODE'] = st.selectbox('Choose Mode', ('Basic', 'Advanced'))

        if toolkit['SUM_MODE'] == 'Basic':
            # FLAGS
            st.markdown('## Options')
            st.markdown('Select one model to use for your NLP Processing. Choose en_core_web_sm for a model that is '
                        'optimised for efficiency or en_core_web_lg for a model that is optimised for accuracy.')
            toolkit['NLP_MODEL'] = st.radio('Select spaCy model', ('en_core_web_sm', 'en_core_web_lg'),
                                            help='Be careful when using the Accuracy Model. Due to caching issues in '
                                                 'the app, changes to the column where data is extracted from in the '
                                                 'file uploaded will reset the portions of the app that follows.')
            if toolkit['NLP_MODEL'] == 'en_core_web_sm':
                try:
                    toolkit['NLP'] = spacy.load('en_core_web_sm')
                except OSError:
                    st.warning('Model not found, downloading...')
                    try:
                        os.system('python -m spacy download en_core_web_sm')
                    except Exception as ex:
                        st.error(f'Unable to download Model. Error: {ex}')
                except Exception as ex:
                    st.error(f'Unknown Error: {ex}. Try again.')
                else:
                    st.info('**Efficiency Model** Loaded!')
            elif toolkit['NLP_MODEL'] == 'en_core_web_lg':
                try:
                    toolkit['NLP'] = spacy.load('en_core_web_lg')
                except OSError:
                    st.warning('Model not found, downloading...')
                    try:
                        os.system('python -m spacy download en_core_web_lg')
                    except Exception as ex:
                        st.error(f'Unable to download Model. Error: {ex}')
                except Exception as ex:
                    st.error(f'Unknown Error: {ex}. Try again.')
                else:
                    st.info('**Accuracy Model** Loaded!')
            toolkit['SENT_LEN'] = st.number_input('Enter the total number of sentences to summarise text to',
                                                  min_value=1,
                                                  max_value=100,
                                                  value=3)
            toolkit['SAVE'] = st.checkbox('Save Outputs?', help='Due to the possibility of files with the same file '
                                                                'name and content being downloaded again, a unique '
                                                                'file identifier is tacked onto the filename.')
            if toolkit['SAVE']:
                if st.checkbox('Override Output Format?'):
                    toolkit['OVERRIDE_FORMAT'] = st.selectbox('Overridden Output Format',
                                                              ('CSV', 'XLSX', 'PKL', 'JSON', 'HDF5'))
                    if toolkit['OVERRIDE_FORMAT'] == toolkit['MODE']:
                        st.warning('Warning: Overridden Format is the same as Input Format')
                else:
                    toolkit['OVERRIDE_FORMAT'] = None

            toolkit['VERBOSE'] = st.checkbox('Display Outputs?')
            if toolkit['VERBOSE']:
                toolkit['VERBOSITY'] = st.slider('Data points',
                                                 key='Data points to display?',
                                                 min_value=0,
                                                 max_value=1000,
                                                 value=20,
                                                 help='Select 0 to display all Data Points')
                toolkit['ADVANCED_ANALYSIS'] = st.checkbox('Display Advanced DataFrame Statistics?',
                                                           help='This option will analyse your DataFrame and display '
                                                                'advanced statistics on it. Note that this will '
                                                                'require some time and processing power to complete. '
                                                                'Deselect this option if this if you do not require '
                                                                'it.')

            # MAIN PROCESSING
            if st.button('Summarise Text', key='runner'):
                if not toolkit['DATA'].empty:
                    try:
                        # CLEAN UP AND STANDARDISE DATAFRAMES
                        toolkit['DATA'] = toolkit['DATA'][[toolkit['DATA_COLUMN']]].astype(str)
                    except KeyError:
                        st.error('Warning: CLEANED CONTENT is not found in the file uploaded. Try again.')
                    except Exception as ex:
                        st.error(ex)
                    else:
                        stopwords = list(STOP_WORDS)
                        pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
                        toolkit['DATA']['SUMMARY'] = toolkit['DATA'][toolkit['DATA_COLUMN']]. \
                            apply(lambda x: summarise(x, stopwords, pos_tag, toolkit['NLP'], toolkit['SENT_LEN']))

                    # SHOW DATASETS
                    if toolkit['VERBOSE']:
                        st.markdown('## Summary DataFrame')
                        printDataFrame(data=toolkit['DATA'], verbose_level=toolkit['VERBOSITY'],
                                       advanced=toolkit['ADVANCED_ANALYSIS'])

                    # SAVE DATA
                    if toolkit['SAVE']:
                        st.markdown('---')
                        st.markdown('## Download Data')
                        if toolkit['OVERRIDE_FORMAT'] is not None:
                            st.markdown(prettyDownload(
                                object_to_download=toolkit['DATA'],
                                download_filename=f'summarised.{toolkit["OVERRIDE_FORMAT"].lower()}',
                                button_text=f'Download Summarised Data',
                                override_index=False,
                                format_=toolkit['OVERRIDE_FORMAT']),
                                unsafe_allow_html=True)
                        else:
                            st.markdown(prettyDownload(object_to_download=toolkit['DATA'],
                                                       download_filename=f'summarised.{toolkit["MODE"].lower()}',
                                                       button_text=f'Download Summarised Data',
                                                       override_index=False,
                                                       format_=toolkit['MODE']),
                                        unsafe_allow_html=True)
                else:
                    st.error('Error: Data not loaded properly. Try again.')

        elif toolkit['SUM_MODE'] == 'Advanced':
            # FLAGS
            st.markdown('## Options')
            st.markdown('Choose the minimum and maximum number of words to summarise to below. If you are an '
                        'advanced user, you may choose to modify the number of input tensors for the model. If '
                        'you do not wish to modify the setting, a default value of 512 will be used for your '
                        'summmary.\n\n'
                        'If your system has a GPU , you may wish to install the GPU (CUDA) enabled version '
                        'of PyTorch. If so, click on the expander below to install the correct version of PyTorch '
                        'and to check if your GPU is enabled.')
            with st.expander('GPU-enabled Features'):
                col, col_ = st.columns(2)
                with col:
                    st.markdown('### PyTorch for CUDA 10.2')
                    if st.button('Install Relevant Packages', key='10.2'):
                        os.system('pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102'
                                  ' -f https://download.pytorch.org/whl/cu102/torch_stable.html')
                with col_:
                    st.markdown('### PyTorch for CUDA 11.3')
                    if st.button('Install Relevant Packages', key='11.3'):
                        os.system('pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113'
                                  ' -f https://download.pytorch.org/whl/cu113/torch_stable.html')
                st.markdown('---')
                if st.button('Check if GPU is properly installed'):
                    st.info(f'GPU Installation Status: **{torch.cuda.is_available()}**')
                if st.button('Check GPU used'):
                    try:
                        st.info(f'GPU Device **{torch.cuda.get_device_name(torch.cuda.current_device())}** in use.')
                    except AssertionError:
                        st.error('Your version of PyTorch is CPU-optimised. Download and install any of the above two '
                                 'supported GPU-enabled PyTorch versions to use your GPU and silence this error.')
                    except Exception as ex:
                        st.error(ex)

            toolkit['SAVE'] = st.checkbox('Save Outputs?')
            if toolkit['SAVE']:
                if st.checkbox('Override Output Format?'):
                    toolkit['OVERRIDE_FORMAT'] = st.selectbox('Overridden Output Format',
                                                              ('CSV', 'XLSX', 'PKL', 'JSON', 'HDF5'))
                    if toolkit['OVERRIDE_FORMAT'] == toolkit['MODE']:
                        st.warning('Warning: Overridden Format is the same as Input Format')
                else:
                    toolkit['OVERRIDE_FORMAT'] = None

            toolkit['VERBOSE'] = st.checkbox('Display Outputs?')

            if toolkit['VERBOSE']:
                toolkit['VERBOSITY'] = st.slider('Data points',
                                                 key='Data points to display?',
                                                 min_value=0,
                                                 max_value=1000,
                                                 value=20,
                                                 help='Select 0 to display all Data Points')
                toolkit['ADVANCED_ANALYSIS'] = st.checkbox('Display Advanced DataFrame Statistics?',
                                                           help='This option will analyse your DataFrame and display '
                                                                'advanced statistics on it. Note that this will '
                                                                'require some time and processing power to complete. '
                                                                'Deselect this option if this if you do not require '
                                                                'it.')

            col2, col2_ = st.columns(2)

            toolkit['MIN_WORDS'] = col2.number_input('Key in the minimum number of words to summarise to',
                                                     min_value=1,
                                                     max_value=1000,
                                                     value=80)
            toolkit['MAX_WORDS'] = col2_.number_input('Key in the maximum number of words to summarise to',
                                                      min_value=80,
                                                      max_value=1000,
                                                      value=150)
            toolkit['MAX_TENSOR'] = st.number_input('Key in the maximum number of vectors to consider',
                                                    min_value=1,
                                                    max_value=10000,
                                                    value=512)

            if st.button('Summarise', key='summary_t5'):
                tokenizer = AutoTokenizer.from_pretrained('t5-base')
                model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)

                if not toolkit['DATA'].empty:
                    # to work with tensors, we need to convert the dataframe to a complex datatype
                    toolkit['DATA'] = toolkit['DATA'].astype(object)
                    toolkit['DATA']['ENCODED'] = toolkit['DATA'][toolkit['DATA_COLUMN']]. \
                        apply(lambda x: tokenizer.encode('summarize: ' + x,
                                                         return_tensors='pt',
                                                         max_length=toolkit['MAX_TENSOR'],
                                                         truncation=True))
                    toolkit['DATA']['OUTPUTS'] = toolkit['DATA']['ENCODED']. \
                        apply(lambda x: model.generate(x,
                                                       max_length=toolkit['MAX_WORDS'],
                                                       min_length=toolkit['MIN_WORDS'],
                                                       length_penalty=5.0,
                                                       num_beams=2))
                    toolkit['DATA']['SUMMARISED'] = toolkit['DATA']['OUTPUTS'].apply(lambda x: tokenizer.decode(x[0]))
                    toolkit['DATA'].drop(columns=['ENCODED', 'OUTPUTS'], inplace=True)
                    toolkit['DATA']['SUMMARISED'] = toolkit['DATA']['SUMMARISED']. \
                        str.replace('<pad> ', '').str.replace('</s>', '')
                    toolkit['DATA'] = toolkit['DATA'].astype(str)

                if toolkit['VERBOSE']:
                    st.markdown('## Summarised Text')
                    printDataFrame(toolkit['DATA'], toolkit['VERBOSITY'], toolkit['ADVANCED_ANALYSIS'])

                if toolkit['SAVE']:
                    st.markdown('---')
                    st.markdown('## Download Summarised Data')
                    if toolkit['OVERRIDE_FORMAT'] is not None:
                        st.markdown(prettyDownload(object_to_download=toolkit['DATA'],
                                                   download_filename=f'summarised.{toolkit["OVERRIDE_FORMAT"].lower()}',
                                                   button_text=f'Download Summarised Data',
                                                   override_index=False,
                                                   format_=toolkit['OVERRIDE_FORMAT']),
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(prettyDownload(object_to_download=toolkit['DATA'],
                                                   download_filename=f'summarised.{toolkit["MODE"].lower()}',
                                                   button_text=f'Download Summarised Data',
                                                   override_index=False,
                                                   format_=toolkit['MODE']),
                                    unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------------------------- #
# |                                              SENTIMENT ANALYSIS                                                  | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif toolkit['APP_MODE'] == 'Analyse Sentiment':
        st.markdown('---')
        st.markdown('# Sentiment Analysis\n'
                    'For this module, both the VADER and TextBlob models will be used to analyse the sentiment of the '
                    'text you upload.\n'
                    'For VADER, your sentiment score will be ')

        # FLAGS
        st.markdown('## Options')
        toolkit['BACKEND_ANALYSER'] = st.selectbox('Choose the Backend Engine Used to Conduct Sentiment Analysis',
                                                   ('VADER', 'TextBlob'),
                                                   help='VADER is more optimised for texts extracted from Social Media '
                                                        'platforms (where slangs and emoticons are used) while '
                                                        'TextBlob performs better for more formal pieces of text. If '
                                                        'you are not sure which to choose, VADER is recommended due to '
                                                        'its higher accuracy of analysis compared to Textblob.')
        toolkit['SAVE'] = st.checkbox('Save Outputs?')
        if toolkit['SAVE']:
            if st.checkbox('Override Output Format?'):
                toolkit['OVERRIDE_FORMAT'] = st.selectbox('Overridden Output Format',
                                                          ('CSV', 'XLSX', 'PKL', 'JSON', 'HDF5'))
                if toolkit['OVERRIDE_FORMAT'] == toolkit['MODE']:
                    st.warning('Warning: Overridden Format is the same as Input Format')
            else:
                toolkit['OVERRIDE_FORMAT'] = None

        toolkit['VERBOSE'] = st.checkbox('Display Outputs?')
        if toolkit['VERBOSE']:
            toolkit['VERBOSITY'] = st.slider('Data points',
                                             key='Data points to display?',
                                             min_value=0,
                                             max_value=1000,
                                             value=20,
                                             help='Select 0 to display all Data Points')
            if toolkit['BACKEND_ANALYSER'] == 'VADER':
                toolkit['COLOUR'] = st.color_picker('Choose Colour of Marker to Display', value='#2ACAEA')
            toolkit['ADVANCED_ANALYSIS'] = st.checkbox('Display Advanced DataFrame Statistics?',
                                                       help='This option will analyse your DataFrame and display '
                                                            'advanced statistics on it. Note that this will require '
                                                            'some time and processing power to complete. Deselect this '
                                                            'option if this if you do not require it.')

        # MAIN PROCESSING
        if st.button('Start Analysis', key='analysis'):
            if not toolkit['DATA'].empty:
                if toolkit['BACKEND_ANALYSER'] == 'VADER':
                    replacer = {
                        r"'": '',
                        r'[^\w\s]': ' ',
                        r' \d+': ' ',
                        r' +': ' '
                    }

                    toolkit['DATA']['VADER SENTIMENT TEXT'] = toolkit['DATA'][toolkit['DATA_COLUMN']]. \
                        replace(to_replace=replacer, regex=True)

                    vader_analyser = SentimentIntensityAnalyzer()
                    sent_score_list = []
                    sent_label_list = []

                    for i in toolkit['DATA']['VADER SENTIMENT TEXT'].tolist():
                        sent_score = vader_analyser.polarity_scores(i)

                        if sent_score['compound'] > 0:
                            sent_score_list.append(sent_score['compound'])
                            sent_label_list.append('Positive')
                        elif sent_score['compound'] == 0:
                            sent_score_list.append(sent_score['compound'])
                            sent_label_list.append('Neutral')
                        elif sent_score['compound'] < 0:
                            sent_score_list.append(sent_score['compound'])
                            sent_label_list.append('Negative')

                    toolkit['DATA']['VADER OVERALL SENTIMENT'] = sent_label_list
                    toolkit['DATA']['VADER OVERALL SCORE'] = sent_score_list
                    toolkit['DATA']['VADER POSITIVE SCORING'] = [vader_analyser.polarity_scores(doc)['pos'] for doc in
                                                                 toolkit['DATA']['VADER SENTIMENT TEXT']
                                                                 .values.tolist()]
                    toolkit['DATA']['VADER NEUTRAL SCORING'] = [vader_analyser.polarity_scores(doc)['neu'] for doc in
                                                                toolkit['DATA']['VADER SENTIMENT TEXT'].values.tolist()]
                    toolkit['DATA']['VADER NEGATIVE SCORING'] = [vader_analyser.polarity_scores(doc)['neg'] for doc in
                                                                 toolkit['DATA']['VADER SENTIMENT TEXT']
                                                                 .values.tolist()]

                elif toolkit['BACKEND_ANALYSER'] == 'TextBlob':
                    try:
                        pol_list = []
                        sub_list = []

                        # APPLY POLARITY FUNCTION
                        toolkit['DATA']['POLARITY SCORE'] = toolkit['DATA'][toolkit['DATA_COLUMN']]. \
                            apply(lambda x: TextBlob(x).sentiment.polarity)
                        for i in toolkit['DATA']['POLARITY SCORE'].tolist():
                            if float(i) > 0:
                                pol_list.append('Positive')
                            elif float(i) < 0:
                                pol_list.append('Negative')
                            elif float(i) == 0:
                                pol_list.append('Neutral')
                        toolkit['DATA']['POLARITY SENTIMENT'] = pol_list

                        # APPLY SUBJECTIVITY FUNCTION
                        toolkit['DATA']['SUBJECTIVITY SCORE'] = toolkit['DATA'][toolkit['DATA_COLUMN']].apply(
                            lambda x: TextBlob(x).sentiment.subjectivity
                        )
                        for i in toolkit['DATA']['SUBJECTIVITY SCORE'].tolist():
                            if float(i) < 0.5:
                                sub_list.append('Objective')
                            elif float(i) > 0.5:
                                sub_list.append('Subjective')
                            elif float(i) == 0.5:
                                sub_list.append('Neutral')
                        toolkit['DATA']['SUBJECTIVITY SENTIMENT'] = sub_list
                    except Exception as ex:
                        st.error(f'Error: {ex}')

                # SHOW DATA
                if toolkit['VERBOSE']:
                    if toolkit['BACKEND_ANALYSER'] == 'VADER':
                        toolkit['HAC_PLOT1'] = None
                        if 'VADER OVERALL SENTIMENT' or 'VADER OVERALL SCORE' in toolkit['DATA'].columns:
                            st.markdown('## Sentiment DataFrame')
                            printDataFrame(data=toolkit['DATA'], verbose_level=toolkit['VERBOSITY'],
                                           advanced=toolkit['ADVANCED_ANALYSIS'])

                            st.markdown('## VADER Score')
                            toolkit['HAC_PLOT'] = ff.create_distplot([toolkit['DATA']['VADER OVERALL SCORE'].tolist()],
                                                                     ['VADER'],
                                                                     colors=[toolkit['COLOUR']],
                                                                     bin_size=0.25,
                                                                     curve_type='normal',
                                                                     show_rug=False,
                                                                     show_hist=False)
                            toolkit['HAC_PLOT'].update_layout(title_text='Distribution Plot',
                                                              xaxis_title='VADER Score',
                                                              yaxis_title='Frequency Density',
                                                              legend_title='Frequency Density')
                            st.plotly_chart(toolkit['HAC_PLOT'], use_container_width=True)
                        else:
                            st.error('Warning: An error is made in the processing of the data. Try again.')

                    elif toolkit['BACKEND_ANALYSER'] == 'TextBlob':
                        if 'POLARITY SCORE' or 'SUBJECTIVITY SCORE' in toolkit['DATA'].columns:
                            st.markdown('## Sentiment DataFrame')
                            printDataFrame(data=toolkit['DATA'], verbose_level=toolkit['VERBOSITY'],
                                           advanced=toolkit['ADVANCED_ANALYSIS'])

                            st.markdown('## Polarity VS Subjectivity')
                            toolkit['HAC_PLOT'] = px.scatter(toolkit['DATA'][['SUBJECTIVITY SCORE', 'POLARITY SCORE']],
                                                             x='SUBJECTIVITY SCORE',
                                                             y='POLARITY SCORE',
                                                             labels={
                                                                 'SUBJECTIVITY SCORE': 'Subjectivity',
                                                                 'POLARITY SCORE': 'Polarity'
                                                             })
                            st.plotly_chart(toolkit['HAC_PLOT'], use_container_width=True)

                            st.markdown('### Normal Distribution Plots of Subjectivity and Polarity')
                            toolkit['HAC_PLOT1'] = ff.create_distplot([toolkit['DATA']['SUBJECTIVITY SCORE'].tolist(),
                                                                       toolkit['DATA']['POLARITY SCORE'].tolist()],
                                                                      ['Subjectivity', 'Polarity'],
                                                                      curve_type='normal',
                                                                      show_rug=False,
                                                                      show_hist=False)
                            st.plotly_chart(toolkit['HAC_PLOT1'], use_container_width=True)
                        else:
                            st.error('Warning: An error is made in the processing of the data. Try again.')

                # SAVE DATA
                if toolkit['SAVE']:
                    st.markdown('---')
                    st.markdown('## Download Data')
                    if toolkit['OVERRIDE_FORMAT'] is not None:
                        st.markdown(prettyDownload(
                            object_to_download=toolkit['DATA'],
                            download_filename=f'sentiment_scores.{toolkit["OVERRIDE_FORMAT"].lower()}',
                            button_text=f'Download Sentiment Score Data',
                            override_index=False,
                            format_=toolkit['OVERRIDE_FORMAT']),
                            unsafe_allow_html=True)
                    else:
                        st.markdown(prettyDownload(object_to_download=toolkit['DATA'],
                                                   download_filename=f'sentiment_scores.{toolkit["MODE"].lower()}',
                                                   button_text=f'Download Sentiment Score Data',
                                                   override_index=False,
                                                   format_=toolkit["MODE"]),
                                    unsafe_allow_html=True)

                    if toolkit['HAC_PLOT'] is not None:
                        st.markdown('## Graphs')
                        st.markdown(prettyDownload(
                            object_to_download=toolkit['HAC_PLOT'],
                            download_filename='plot.png',
                            button_text=f'Download Plot',
                            override_index=False),
                            unsafe_allow_html=True
                        )
                    if toolkit['HAC_PLOT1'] is not None:
                        st.markdown(prettyDownload(
                            object_to_download=toolkit['HAC_PLOT1'],
                            download_filename='normal_plot.png',
                            button_text=f'Download Normal Distribution Data',
                            override_index=False),
                            unsafe_allow_html=True
                        )
            else:
                st.error('Error: Data not loaded properly. Try again.')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                TOPIC MODELLING                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif toolkit['APP_MODE'] == 'Topic Modelling':
        st.markdown('---')
        st.markdown('# Topic Modelling')
        st.markdown('Ensure that your data is **lemmatized and properly cleaned**; data **should not be tokenized** '
                    'for this step. Use the Load, Clean and Visualise module to clean and lemmatize your data if you '
                    'have not done so already.\n\n')
        with st.expander('Short Explanation on Models used'):
            st.markdown(
                '#### Latent Dirichlet Allocation (LDA)\n'
                'Extracted from '
                'https://towardsdatascience.com/topic-modeling-quora-questions-with-lda-nmf-aff8dce5e1dd, LDA is '
                'the Model used for finding all the hidden probability distributions of a certain set of data, '
                'which would allow us to discover hidden groupings of data points within a set of data.\n\n'
                '#### Non-Negative Matrix Factorization (NMF)\n'
                'Extracted from '
                'https://medium.com/analytics-vidhya/topic-modeling-with-non-negative-matrix-factorization-nmf'
                '-3caf3a6bb6da, NMF is an unsupervised learning technique to decompose (factorise) high-'
                'dimensional vectors into non-negative lower-dimensional vectors.\n\n'
                '#### Latent Semantic Indexing (LSI)\n'
                'Extracted from https://www.searchenginejournal.com/latent-semantic-indexing-wont-help-seo/240705/'
                '#:~:text=Latent%20semantic%20indexing%20(also%20referred,of%20those%20words%20and%20documents, '
                'LSI is a technique of analysing a document to discover statistical co-occurrences of words which '
                'appear to gather, which gives us insights into the topics of the words and of the document.')

        st.markdown('## Topic Modelling Model Selection')
        toolkit['NLP_TOPIC_MODEL'] = st.selectbox('Choose Model to use', ('Latent Dirichlet Allocation',
                                                                          'Non-Negative Matrix Factorization',
                                                                          'Latent Semantic Indexing'))
        st.info(f'**{toolkit["NLP_TOPIC_MODEL"]}** Selected')

        # FLAGS
        st.markdown('## Options')
        toolkit['SAVE'] = st.checkbox('Save Outputs?')
        if toolkit['SAVE']:
            if st.checkbox('Override Output Format?'):
                toolkit['OVERRIDE_FORMAT'] = st.selectbox('Overridden Output Format',
                                                          ('CSV', 'XLSX', 'PKL', 'JSON', 'HDF5'))
                if toolkit['OVERRIDE_FORMAT'] == toolkit['MODE']:
                    st.warning('Warning: Overridden Format is the same as Input Format')
            else:
                toolkit['OVERRIDE_FORMAT'] = None

        toolkit['VERBOSE'] = st.checkbox('Display Outputs?', help='Note: For LSI, if verbose is enabled, the model '
                                                                  'will be refitted such that only 2 components are '
                                                                  'considered. If you wish to turn off this feature, '
                                                                  'deselect the "Generate Plot" selection below.')
        if toolkit['VERBOSE']:
            toolkit['VERBOSITY'] = st.slider('Data points',
                                             key='Data points to display?',
                                             min_value=0,
                                             max_value=1000,
                                             value=20,
                                             help='Select 0 to display all Data Points')
            toolkit['ADVANCED_ANALYSIS'] = st.checkbox('Display Advanced DataFrame Statistics?',
                                                       help='This option will analyse your DataFrame and display '
                                                            'advanced statistics on it. Note that this will require '
                                                            'some time and processing power to complete. Deselect this '
                                                            'option if this if you do not require it.')

        col1, col2 = st.columns(2)

        toolkit['NUM_TOPICS'] = col1.number_input('Number of Topics to Generate',
                                                  min_value=1,
                                                  max_value=100,
                                                  step=1,
                                                  value=10)
        toolkit['MAX_FEATURES'] = col2.number_input('Maximum Number of Features to Extract',
                                                    min_value=1,
                                                    max_value=99999,
                                                    step=1,
                                                    value=5000)

        toolkit['MAX_ITER'] = col1.number_input('Iterations of Model Training (Epochs)',
                                                min_value=1,
                                                max_value=10000,
                                                step=1,
                                                value=10)

        if toolkit['NLP_TOPIC_MODEL'] == 'Latent Dirichlet Allocation':
            toolkit['MIN_DF'] = col2.number_input('Minimum (Cardinal) Frequency of Words to Consider',
                                                  min_value=1,
                                                  max_value=100,
                                                  step=1,
                                                  value=5)
            toolkit['MAX_DF'] = col1.number_input('Maximum (Percentage) Frequency of Words to Consider',
                                                  min_value=0.,
                                                  max_value=1.,
                                                  step=0.01,
                                                  format='%.2f',
                                                  value=0.90)
            toolkit['WORKER'] = col2.number_input('Number of CPU Cores to Use',
                                                  min_value=1,
                                                  max_value=multiprocessing.cpu_count(),
                                                  step=1,
                                                  value=1,
                                                  help='The number of effective cores is calculated from multiplying '
                                                       'the number of cores on your machine and the number of threads '
                                                       'in your CPU.')

        elif toolkit['NLP_TOPIC_MODEL'] == 'Non-Negative Matrix Factorization':
            toolkit['MIN_DF'] = col2.number_input('Minimum (Cardinal) Frequency of Words to Consider',
                                                  min_value=1,
                                                  max_value=100,
                                                  step=1,
                                                  value=2)
            toolkit['MAX_DF'] = col1.number_input('Maximum (Percentage) Frequency of Words to Consider',
                                                  min_value=0.,
                                                  max_value=1.,
                                                  step=0.01,
                                                  format='%.2f',
                                                  value=0.95)
            toolkit['ALPHA'] = col2.number_input('Alpha Value for NMF Model',
                                                 min_value=0.,
                                                 max_value=1.,
                                                 step=0.01,
                                                 format='%.2f',
                                                 value=.1,
                                                 help='Constant that multiplies the regularization terms. Set it to '
                                                      'zero to have no regularization. Regularization is not scaled by '
                                                      'the number of features to consider in the model.')
            toolkit['L1_RATIO'] = col1.number_input('L1 Ratio for NMF Model',
                                                    min_value=0.,
                                                    max_value=1.,
                                                    step=0.01,
                                                    format='%.2f',
                                                    value=.5,
                                                    help='The regularization mixing parameter, with 0 <= l1_ratio <= 1'
                                                         '.')

        elif toolkit['NLP_TOPIC_MODEL'] == 'Latent Semantic Indexing':
            if toolkit['VERBOSE']:
                toolkit['PLOT'] = st.checkbox('Generate LSI Plot?')
                if toolkit['PLOT']:
                    toolkit['W_PLOT'] = st.checkbox('Generate Word Representation of LSI Plot?')
                    toolkit['COLOUR'] = st.color_picker('Choose Colour of Marker to Display', value='#2ACAEA')

        if st.button('Start Modelling', key='topic'):
            if not toolkit['DATA'].empty:
                try:
                    toolkit['CV'] = CountVectorizer(min_df=toolkit['MIN_DF'],
                                                    max_df=toolkit['MAX_DF'],
                                                    stop_words='english',
                                                    lowercase=True,
                                                    token_pattern=r'[a-zA-Z\-][a-zA-Z\-]{2,}',
                                                    max_features=toolkit['MAX_FEATURES'])
                    toolkit['VECTORISED'] = toolkit['CV'].fit_transform(toolkit['DATA'][toolkit['DATA_COLUMN']])
                except ValueError:
                    st.error('Error: The column loaded is empty or has invalid data points. Try again.')
                except Exception as ex:
                    st.error(f'Error: {ex}')
                else:
                    # LDA
                    if toolkit['NLP_TOPIC_MODEL'] == 'Latent Dirichlet Allocation':
                        toolkit['LDA_MODEL'] = LatentDirichletAllocation(n_components=toolkit['NUM_TOPICS'],
                                                                         max_iter=toolkit['MAX_ITER'],
                                                                         learning_method='online',
                                                                         n_jobs=toolkit['WORKER'])
                        toolkit['LDA_DATA'] = toolkit['LDA_MODEL'].fit_transform(toolkit['VECTORISED'])

                        if toolkit['VERBOSE']:
                            st.markdown('## Model Data')
                            toolkit['TOPIC_TEXT'] = modelIterator(toolkit['LDA_MODEL'], toolkit['CV'],
                                                                  top_n=toolkit['NUM_TOPICS'])
                        else:
                            toolkit['TOPIC_TEXT'] = modelIterator(toolkit['LDA_MODEL'], toolkit['CV'],
                                                                  top_n=toolkit['NUM_TOPICS'], vb=False)

                        toolkit['KW'] = pd.DataFrame(dominantTopic(vect=toolkit['CV'], model=toolkit['LDA_MODEL'],
                                                                   n_words=toolkit['NUM_TOPICS']))
                        toolkit['KW'].columns = [f'word_{i}' for i in range(toolkit["KW"].shape[1])]
                        toolkit['KW'].index = [f'topic_{i}' for i in range(toolkit["KW"].shape[0])]

                        # THIS VISUALISES ALL THE DOCUMENTS IN THE DATASET PROVIDED
                        toolkit['LDA_VIS'] = pyLDAvis.sklearn.prepare(toolkit['LDA_MODEL'], toolkit['VECTORISED'],
                                                                      toolkit['CV'], mds='tsne')

                        if toolkit['VERBOSE']:
                            st.markdown('## Topic Label\n'
                                        'The following frame will show you the words that are associated with a '
                                        'certain topic.')
                            printDataFrame(data=toolkit['KW'], verbose_level=toolkit['NUM_TOPICS'],
                                           advanced=toolkit['ADVANCED_ANALYSIS'])

                            st.markdown('## LDA\n'
                                        f'The following HTML render displays the top {toolkit["NUM_TOPICS"]} of '
                                        f'Topics generated from all the text provided in your dataset.')
                            toolkit['LDA_VIS_STR'] = pyLDAvis.prepared_data_to_html(toolkit['LDA_VIS'])
                            streamlit.components.v1.html(toolkit['LDA_VIS_STR'], width=1300, height=800)

                        if toolkit['SAVE']:
                            st.markdown('---')
                            st.markdown('## Save Data\n'
                                        '### Topics')
                            for i in range(len(toolkit['TOPIC_TEXT'])):
                                if toolkit['OVERRIDE_FORMAT'] is not None:
                                    st.markdown(prettyDownload(
                                        object_to_download=toolkit['TOPIC_TEXT'][i],
                                        download_filename=f'lda_topics_{i}.{toolkit["OVERRIDE_FORMAT"].lower()}',
                                        button_text=f'Download Topic List Data Entry {i}',
                                        override_index=False,
                                        format_=toolkit['OVERRIDE_FORMAT']),
                                        unsafe_allow_html=True)
                                else:
                                    st.markdown(prettyDownload(
                                        object_to_download=toolkit['TOPIC_TEXT'][i],
                                        download_filename=f'lda_topics_{i}.{toolkit["MODE"].lower()}',
                                        button_text=f'Download Topic List Data Entry {i}',
                                        override_index=False,
                                        format_=toolkit['MODE']),
                                        unsafe_allow_html=True)

                            st.markdown('### Topic/Word List')
                            if toolkit['OVERRIDE_FORMAT'] is not None:
                                st.markdown(prettyDownload(
                                    object_to_download=toolkit['KW'],
                                    download_filename=f'summary_topics.{toolkit["OVERRIDE_FORMAT"].lower()}',
                                    button_text=f'Download Summarised Topic/Word Data',
                                    override_index=False,
                                    format_=toolkit['OVERRIDE_FORMAT']),
                                    unsafe_allow_html=True)
                            else:
                                st.markdown(prettyDownload(
                                    object_to_download=toolkit['KW'],
                                    download_filename=f'summary_topics.{toolkit["MODE"].lower()}',
                                    button_text=f'Download Summarised Topic/Word Data',
                                    override_index=False,
                                    format_=toolkit['MODE']),
                                    unsafe_allow_html=True)

                            st.markdown('### Other Requested Data')
                            st.markdown(prettyDownload(
                                object_to_download=pyLDAvis.prepared_data_to_html(toolkit['LDA_VIS']),
                                download_filename='lda.html',
                                button_text='Download pyLDAvis Rendering',
                                override_index=False),
                                unsafe_allow_html=True
                            )

                    # NMF
                    elif toolkit['NLP_TOPIC_MODEL'] == 'Non-Negative Matrix Factorization':
                        toolkit['TFIDF_MODEL'] = TfidfVectorizer(max_df=toolkit['MAX_DF'],
                                                                 min_df=toolkit['MIN_DF'],
                                                                 max_features=toolkit['MAX_FEATURES'],
                                                                 stop_words='english')
                        toolkit['TFIDF_VECTORISED'] = toolkit['TFIDF_MODEL'].fit_transform(toolkit['DATA']
                                                                                           [toolkit['DATA_COLUMN']]
                                                                                           .values.astype(str))
                        toolkit['NMF_MODEL'] = NMF(n_components=toolkit['NUM_TOPICS'],
                                                   max_iter=toolkit['MAX_ITER'],
                                                   random_state=1,
                                                   alpha=toolkit['ALPHA'],
                                                   l1_ratio=toolkit['L1_RATIO']).fit(toolkit['TFIDF_VECTORISED'])

                        if toolkit['VERBOSE']:
                            st.markdown('## Model Data')
                            toolkit['TOPIC_TEXT'] = modelIterator(toolkit['NMF_MODEL'], toolkit['TFIDF_MODEL'],
                                                                  top_n=toolkit['NUM_TOPICS'])
                        else:
                            toolkit['TOPIC_TEXT'] = modelIterator(model=toolkit['NMF_MODEL'],
                                                                  vectoriser=toolkit['TFIDF_MODEL'],
                                                                  top_n=toolkit['NUM_TOPICS'],
                                                                  vb=False)

                        toolkit['KW'] = pd.DataFrame(dominantTopic(model=toolkit['NMF_MODEL'],
                                                                   vect=toolkit['TFIDF_MODEL'],
                                                                   n_words=toolkit['NUM_TOPICS']))
                        toolkit['KW'].columns = [f'word_{i}' for i in range(toolkit['KW'].shape[1])]
                        toolkit['KW'].index = [f'topic_{i}' for i in range(toolkit['KW'].shape[0])]

                        if toolkit['VERBOSE']:
                            st.markdown('## NMF Topic DataFrame')
                            printDataFrame(data=toolkit['KW'],
                                           verbose_level=toolkit['NUM_TOPICS'],
                                           advanced=toolkit['ADVANCED_ANALYSIS'])

                        if toolkit['SAVE']:
                            st.markdown('---')
                            st.markdown('## Save Data\n'
                                        '### Topics')
                            for i in range(len(toolkit['TOPIC_TEXT'])):
                                if toolkit['OVERRIDE_FORMAT'] is not None:
                                    st.markdown(prettyDownload(
                                        object_to_download=toolkit['TOPIC_TEXT'][i],
                                        download_filename=f'nmf_topics_{i}.{toolkit["OVERRIDE_FORMAT"].lower()}',
                                        button_text=f'Download Topic List Data Entry {i}',
                                        override_index=False,
                                        format_=toolkit['OVERRIDE_FORMAT']),
                                        unsafe_allow_html=True)
                                else:
                                    st.markdown(prettyDownload(
                                        object_to_download=toolkit['TOPIC_TEXT'][i],
                                        download_filename=f'nmf_topics_{i}.{toolkit["MODE"].lower()}',
                                        button_text=f'Download Topic List Data Entry {i}',
                                        override_index=False,
                                        format_=toolkit['MODE']),
                                        unsafe_allow_html=True)

                            st.markdown('### Topic/Word List')
                            if toolkit['OVERRIDE_FORMAT'] is not None:
                                st.markdown(prettyDownload(
                                    object_to_download=toolkit['KW'],
                                    download_filename=f'summary_topics.{toolkit["OVERRIDE_FORMAT"].lower()}',
                                    button_text=f'Download Summarised Topic/Word Data',
                                    override_index=False,
                                    format_=toolkit['OVERRIDE_FORMAT']),
                                    unsafe_allow_html=True)
                            else:
                                st.markdown(prettyDownload(
                                    object_to_download=toolkit['KW'],
                                    download_filename=f'summary_topics.{toolkit["MODE"].lower()}',
                                    button_text=f'Download Summarised Topic/Word Data',
                                    override_index=False,
                                    format_=toolkit['MODE']),
                                    unsafe_allow_html=True)

                    # LSI
                    elif toolkit['NLP_TOPIC_MODEL'] == 'Latent Semantic Indexing':
                        toolkit['LSI_MODEL'] = TruncatedSVD(n_components=toolkit['NUM_TOPICS'],
                                                            n_iter=toolkit['MAX_ITER'])
                        toolkit['LSI_DATA'] = toolkit['LSI_MODEL'].fit_transform(toolkit['VECTORISED'])

                        if toolkit['VERBOSE']:
                            st.markdown('## Model Data')
                            toolkit['TOPIC_TEXT'] = modelIterator(toolkit['LSI_MODEL'], toolkit['CV'],
                                                                  top_n=toolkit['NUM_TOPICS'])
                        else:
                            toolkit['TOPIC_TEXT'] = modelIterator(toolkit['LSI_MODEL'], toolkit['CV'],
                                                                  top_n=toolkit['NUM_TOPICS'], vb=False)

                        toolkit['KW'] = pd.DataFrame(dominantTopic(model=toolkit['LSI_MODEL'], vect=toolkit['CV'],
                                                                   n_words=toolkit['NUM_TOPICS']))
                        toolkit['KW'].columns = [f'word_{i}' for i in range(toolkit['KW'].shape[1])]
                        toolkit['KW'].index = [f'topic_{i}' for i in range(toolkit['KW'].shape[0])]

                        if toolkit['VERBOSE']:
                            st.markdown('## LSI Topic DataFrame')
                            printDataFrame(data=toolkit['KW'],
                                           verbose_level=toolkit['VERBOSITY'],
                                           advanced=toolkit['ADVANCED_ANALYSIS'])

                            if toolkit['PLOT']:
                                st.markdown('## LSI(SVD) Scatterplot\n'
                                            'Note that the following visualisation will use a Topic count of 2, '
                                            'overriding previous inputs above, as you are only able to visualise data '
                                            'on 2 axis. However the full analysis result of the above operation will '
                                            'be saved in the dataset you provided and be available for download later '
                                            'on in the app.\n\n'
                                            'The main aim of the scatterplot is to show the similarity between topics, '
                                            'which is measured by the distance between markers as shown in the '
                                            'following diagram. The diagram contained within the expander is the same '
                                            'as the marker diagram, just that the markers are all replaced by the '
                                            'topic words the markers actually represent.')
                                svd_2d = TruncatedSVD(n_components=2)
                                data_2d = svd_2d.fit_transform(toolkit['VECTORISED'])

                                toolkit['MAR_FIG'] = go.Scattergl(
                                    x=data_2d[:, 0],
                                    y=data_2d[:, 1],
                                    mode='markers',
                                    marker=dict(
                                        color=toolkit['COLOUR'],
                                        line=dict(width=1)
                                    ),
                                    text=toolkit['CV'].get_feature_names(),
                                    hovertext=toolkit['CV'].get_feature_names(),
                                    hoverinfo='text'
                                )
                                toolkit['MAR_FIG'] = [toolkit['MAR_FIG']]
                                toolkit['MAR_FIG'] = go.Figure(data=toolkit['MAR_FIG'],
                                                               layout=go.Layout(title='Scatter Plot'))
                                st.plotly_chart(toolkit['MAR_FIG'])

                                if toolkit['W_PLOT']:
                                    with st.expander('Show Word Plots'):
                                        toolkit['WORD_FIG'] = go.Scattergl(
                                            x=data_2d[:, 0],
                                            y=data_2d[:, 1],
                                            mode='text',
                                            marker=dict(
                                                color=toolkit['COLOUR'],
                                                line=dict(width=1)
                                            ),
                                            text=toolkit['CV'].get_feature_names(),
                                        )
                                        toolkit['WORD_FIG'] = [toolkit['WORD_FIG']]
                                        toolkit['WORD_FIG'] = go.Figure(data=toolkit['WORD_FIG'],
                                                                        layout=go.Layout(title='Scatter Word Plot'))
                                        st.plotly_chart(toolkit['WORD_FIG'])

                        if toolkit['SAVE']:
                            st.markdown('---')
                            st.markdown('## Save Data\n'
                                        '### Topics')
                            for i in range(len(toolkit['TOPIC_TEXT'])):
                                if toolkit['OVERRIDE_FORMAT'] is not None:
                                    st.markdown(prettyDownload(
                                        object_to_download=toolkit['TOPIC_TEXT'][i],
                                        download_filename=f'lsi_topics_{i}.{toolkit["OVERRIDE_FORMAT"].lower()}',
                                        button_text=f'Download Topic List Data Entry {i}',
                                        override_index=False,
                                        format_=toolkit['OVERRIDE_FORMAT']),
                                        unsafe_allow_html=True)
                                else:
                                    st.markdown(prettyDownload(
                                        object_to_download=toolkit['TOPIC_TEXT'][i],
                                        download_filename=f'lsi_topics_{i}.{toolkit["MODE"].lower()}',
                                        button_text=f'Download Topic List Data Entry {i}',
                                        override_index=False,
                                        format_=toolkit['MODE']),
                                        unsafe_allow_html=True)

                            st.markdown('### Topic/Word List')
                            if toolkit['OVERRIDE_FORMAT'] is not None:
                                st.markdown(prettyDownload(
                                    object_to_download=toolkit['KW'],
                                    download_filename=f'summary_topics.{toolkit["OVERRIDE_FORMAT"].lower()}',
                                    button_text=f'Download Summarised Topic/Word Data',
                                    override_index=False,
                                    format_=toolkit['OVERRIDE_FORMAT']),
                                    unsafe_allow_html=True)
                            else:
                                st.markdown(prettyDownload(
                                    object_to_download=toolkit['KW'],
                                    download_filename=f'summary_topics.{toolkit["MODE"].lower()}',
                                    button_text=f'Download Summarised Topic/Word Data',
                                    override_index=False,
                                    format_=toolkit['MODE']),
                                    unsafe_allow_html=True)

                            if toolkit['VERBOSE'] and toolkit['PLOT']:
                                st.markdown('### Other Requested Data')
                                st.markdown(prettyDownload(
                                    object_to_download=toolkit['MAR_FIG'],
                                    download_filename=f'marker_figure.png',
                                    button_text='Download Marked Figure',
                                    override_index=False),
                                    unsafe_allow_html=True)

                                if toolkit['W_PLOT']:
                                    st.markdown(prettyDownload(
                                        object_to_download=toolkit['WORD_FIG'],
                                        download_filename=f'word_figure.png',
                                        button_text='Download Word Figure',
                                        override_index=False),
                                        unsafe_allow_html=True)
            else:
                st.error('Error: File not loaded properly. Try again.')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                              TOPIC CLASSIFICATION                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
    elif toolkit['APP_MODE'] == 'Topic Classification':
        st.markdown('---')
        st.markdown('# Topic Classification')
        st.markdown('This function expands on the Topic Modelling function within the NLP Toolkit, but allowing users '
                    'to use pretrained ML models to classify the news articles into the list of topics you input '
                    'into the app. For nw, only the Zero-Shot Classification model from Huggingface is implemented.\n\n'
                    'This function performs best when GPU is enabled. To enable your GPU to run the classification '
                    'process, click on the following expander and download and install the required packages.')
        with st.expander('GPU-enabled Features'):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('### PyTorch for CUDA 10.2')
                if st.button('Install Relevant Packages', key='10.2'):
                    os.system('pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102'
                              ' -f https://download.pytorch.org/whl/cu102/torch_stable.html')
            with col2:
                st.markdown('### PyTorch for CUDA 11.3')
                if st.button('Install Relevant Packages', key='11.3'):
                    os.system('pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113'
                              ' -f https://download.pytorch.org/whl/cu113/torch_stable.html')
            st.markdown('---')
            if st.button('Check if GPU is properly installed'):
                st.info(f'GPU Installation Status: **{torch.cuda.is_available()}**')
            if st.button('Check GPU used'):
                try:
                    st.info(f'GPU Device **{torch.cuda.get_device_name(torch.cuda.current_device())}** in use.')
                except AssertionError:
                    st.error('Your version of PyTorch is CPU-optimised. Download and install any of the above two '
                             'supported GPU-enabled PyTorch versions to use your GPU and silence this error.')
                except Exception as ex:
                    st.error(ex)

        st.markdown('## Options')
        toolkit['SAVE'] = st.checkbox('Save Outputs?', help='Due to the possibility of files with the same file name '
                                                            'and content being downloaded again, a unique file '
                                                            'identifier is tacked onto the filename.')
        toolkit['VERBOSE'] = st.checkbox('Display Outputs?')

        if toolkit['VERBOSE']:
            toolkit['VERBOSITY'] = st.slider('Data points',
                                             key='Data points to display?',
                                             min_value=0,
                                             max_value=1000,
                                             value=20,
                                             help='Select 0 to display all Data Points')
            toolkit['ADVANCED_ANALYSIS'] = st.checkbox('Display Advanced DataFrame Statistics?',
                                                       help='This option will analyse your DataFrame and display '
                                                            'advanced statistics on it. Note that this will require '
                                                            'some time and processing power to complete. Deselect this '
                                                            'option if this if you do not require it.')
        toolkit['CLASSIFY_TOPIC'] = st_tags(label='**Topics**',
                                            text='Press Enter to extend list...',
                                            maxtags=9999999,
                                            key='classify_topics')

        if len(toolkit['CLASSIFY_TOPIC']) != 0:
            st.info(f'**{toolkit["CLASSIFY_TOPIC"]}** Topics are Detected!')
        else:
            st.info('No Topics Detected.')

        if st.button('Classify Text', key='classify'):
            toolkit['DATA'] = toolkit['DATA'].astype(object)
            classifier = pipeline('zero-shot-classification')
            toolkit['DATA']['TEST'] = toolkit['DATA'][toolkit['DATA_COLUMN']]. \
                apply(lambda x: classifier(x, toolkit['CLASSIFY_TOPIC']))
            toolkit['DATA']['CLASSIFIED'] = toolkit['DATA']['TEST']. \
                apply(lambda x: list(zip(x['labels'].tolist(), x['scores'].tolist())))
            toolkit['DATA']['MOST PROBABLE TOPIC'] = toolkit['DATA']['CLASSIFIED']. \
                apply(lambda x: max(x, key=itemgetter[1])[0])
            toolkit['DATA'] = toolkit['DATA'].astype(str)

            if toolkit['VERBOSE']:
                st.markdown('## Classified Data')
                printDataFrame(data=toolkit['DATA'], verbose_level=toolkit['VERBOSITY'],
                               advanced=toolkit['ADVANCED_ANALYSIS'])

            if toolkit['SAVE']:
                st.markdown('---')
                st.markdown('## Download Data')
                if toolkit['OVERRIDE_FORMAT'] is not None:
                    st.markdown(prettyDownload(
                        object_to_download=toolkit['DATA'],
                        download_filename=f'classified.{toolkit["OVERRIDE_FORMAT"].lower()}',
                        button_text=f'Download Classified Data',
                        override_index=False,
                        format_=toolkit['OVERRIDE_FORMAT']),
                        unsafe_allow_html=True)
                else:
                    st.markdown(prettyDownload(
                        object_to_download=toolkit['DATA'],
                        download_filename=f'classified.{toolkit["MODE"].lower()}',
                        button_text=f'Download Classified Data',
                        override_index=False,
                        format_=toolkit['MODE']),
                        unsafe_allow_html=True)
