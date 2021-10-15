"""
Generates a helper class to assist with the generation of multiple Streamlit apps through object-oriented programming
"""

# IMPORT STREAMLIT
import numpy as np
import streamlit as st
import pathlib
import pandas as pd

STREAMLIT_STATIC_PATH = ''
DOWNLOAD_PATH = ''


# DEFINE THE MULTIPAGE CLASS TO MANAGE THE APPS
class MultiPage:
    """
    Combines and manages the different modules within the streamlit application
    """

    global STREAMLIT_STATIC_PATH, DOWNLOAD_PATH

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []

    def add_page(self, title, func) -> None:
        """
        Class Method to Add pages to the project

        Arguments
        ----------
        title ([str]):          The title of page which we are adding to the list of apps
        func:                   Python function to render this page in Streamlit
        ----------
        """

        self.pages.append({"title": title,
                           "function": func
                           })

    def run(self):
        """
        Dropdown menu to select the page to run
        """
        with st.sidebar.container():
            # DEFINE THE TITLE AND CONTENTS OF THE MAIN PAGE
            st.title('News Analyser')
            st.markdown('This web app was created for the purposes of the SPF Coding Challenge to analyse news '
                        'articles and provide tools to make sense of the news articles loaded into the app.\n\n')
            st.markdown('## Project Contributors: \n\n'
                        '* Ong Jung Yi, PNSD\n\n'
                        '* George Tay, PNSD\n\n')
            st.markdown('## App Modules\n'
                        'Select the following available modules implemented:')

        # PAGE SELECTOR
        page = st.sidebar.selectbox('NLP Functions',
                                    self.pages,
                                    format_func=lambda page: page['title'])
        # RUN THE APP
        page['function']()
