"""
This file contains all the default parameters used in the app
"""
import os
import typing

import pandas as pd
import nltk
import pathlib
import streamlit as st
from texthero import preprocessing
from datetime import datetime

# STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
# DOWNLOAD_PATH = (STREAMLIT_STATIC_PATH / "downloads")

# download the corpus
nltk.download('words')

load_clean_visualise = {
    'FILE': 'Small File(s)',
    'MODE': 'CSV',
    'DATA_PATH': None,
    'DATA': pd.DataFrame(),
    'CSP': None,
    'CLEAN': False,
    'CLEAN_MODE': 'Simple',
    'SAVE': False,
    'OVERRIDE_FORMAT': None,
    'VERBOSE': False,
    'VERBOSITY': False,
    'CLEANED_DATA': pd.DataFrame(),
    'CLEANED_DATA_TOKENIZED': pd.DataFrame(),
    'ADVANCED_ANALYSIS': False,
    'SIMPLE_PIPELINE': [
        preprocessing.remove_html_tags,
        preprocessing.remove_diacritics,
        preprocessing.remove_whitespace,
        preprocessing.remove_urls,
        preprocessing.drop_no_content
    ],
    'PIPELINE': [
        preprocessing.fillna,
        preprocessing.lowercase,
        preprocessing.remove_punctuation,
        preprocessing.remove_html_tags,
        preprocessing.remove_diacritics,
        preprocessing.remove_whitespace,
        preprocessing.remove_urls,
        preprocessing.drop_no_content
    ],
    # APPLY preprocessing.remove_digits(only_blocks=False) TO THE CUSTOM PIPELINE AFTER CLEANING
    'FINALISED_DATA_LIST': [],
    'DATA_COLUMN': None,
    'TOKENIZE': True,
    'EXTEND_STOPWORD': False,
    'STOPWORD_LIST': '',
    'ENGLISH_WORDS': set(nltk.corpus.words.words()),
    'FINALISE': False,
    'ANALYSIS_MODE': 'Data Cleaning',
    'WORLD_MAP': True,
    'GLOBE_DATA': None,
    'GLOBE_FIG': None,
    'MATCH': False,
    'QUERY': None,
    'QUERY_SUCCESS': False,
    'QUERY_MODE': None,
    'QUERY_DATA': pd.DataFrame(),
    'MOD_MODE': 'Country Extraction',
    'FIXED_KEY': True,
    'HEIGHT': 400,
}

dtm = {
    'DTM': pd.DataFrame(),
    'FILE': 'Small File(s)',
    'DATA': pd.DataFrame(),
    'DATA_PATH': None,
    'ANALYSIS': False,
    'VERBOSE_DTM': False,
    'VERBOSITY_DTM': False,
    'VERBOSE_ANALYSIS': False,
    'SAVE': False,
    'OVERRIDE_FORMAT': None,
    'MODE': 'CSV',
    'DTM_copy': pd.DataFrame(),
    'N': 100,
    'CSP': None,
    'ADVANCED_ANALYSIS': False,
    'FINALISED_DATA_LIST': [],
    'DATA_COLUMN': None,
    'TOP_N_WORD_FIG': None
}

toolkit = {
    'DATA': pd.DataFrame(),
    'FILE': 'Small File(s)',
    'MODE': 'CSV',
    'DATA_PATH': None,
    'CSP': None,
    'SAVE': False,
    'OVERRIDE_FORMAT': None,
    'VERBOSE': False,
    'VERBOSITY': 20,
    'APP_MODE': 'Wordcloud',
    'BACKEND_ANALYSER': 'VADER',
    'MAX_WORDS': 200,
    'CONTOUR_WIDTH': 3,
    'HEIGHT': 400,
    'WIDTH': 800,
    'SENT_LEN': 3,
    'NUM_TOPICS': 10,
    'TOPIC_FRAME': None,
    'LDA_VIS': None,
    'LDA_MODEL': None,
    'KW': None,
    'TFIDF_MODEL': None,
    'TFIDF_VECTORISED': None,
    'NMF_MODEL': None,
    'LSI_MODEL': None,
    'LSI_DATA': None,
    'MAR_FIG': None,
    'WORD_FIG': None,
    'LDA_VIS_STR': None,
    'LDA_DATA': None,
    'MODEL': None,
    'ADVANCED_ANALYSIS': False,
    'NLP_MODEL': 'en_core_web_sm',
    'DATA_COLUMN': None,
    'NLP': None,
    'ONE_DATAPOINT': False,
    'DATAPOINT_SELECTOR': 0,
    'NLP_TOPIC_MODEL': 'Latent Dirichlet Allocation',
    'MIN_DF': 2,
    'MAX_DF': 0.95,
    'MAX_ITER': 100,
    'CV': None,
    'VECTORISED': None,
    'COLOUR': None,
    'COLOUR_BCKGD': None,
    'COLOUR_TXT': None,
    'TOPIC_TEXT': [],
    'SVG': None,
    'HAC_PLOT': None,
    'HAC_PLOT1': None,
    'WORKER': 1,
    'MAX_FEATURES': 5000,
    'ALPHA': 0.1,
    'L1_RATIO': 0.5,
    'PLOT': False,
    'W_PLOT': False,
    'MIN_WORDS': 80,
    'SUM_MODE': 'Basic'
}
