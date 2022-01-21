"""This module runs when the pyfiles apps are imported, downloading the corpora needed in this app"""

import streamlit as st
import nltk

with st.spinner('Downloading WordNet Corpora...'):
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
