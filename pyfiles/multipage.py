"""
Generates a helper class to assist with the generation of multiple Streamlit apps through object-oriented programming
"""

# IMPORT STREAMLIT
import numpy as np
import streamlit as st
import pathlib
import pandas as pd

# STATIC DOWNLOAD FILE PATHS
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
DOWNLOAD_PATH = (STREAMLIT_STATIC_PATH / "downloads")
if not DOWNLOAD_PATH.is_dir():
    DOWNLOAD_PATH.mkdir()


# DEFINE THE MULTIPAGE CLASS TO MANAGE THE APPS
class MultiPage:
    """
    Combines and manages the streamlit apps
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
                        'articles and provide tools to make sense of the news articles loaded into the app.\n\n'
                        'For all modules in the app, kindly use the following template CSV file to store and '
                        'load up your data. As many functions of the app relies on the format of the template '
                        'file, failure to replicate the template in your raw data or failure to use the template '
                        'may result in unexpected errors.')

            st.markdown('## Base Template')
            frame0 = {
                'CONTENT': [
                    'Key in your data in this column',
                    'And extend the index column (left of this column) to the length of your data',
                    'You may safely delete the preloaded data in this column'
                ]
            }
            frame1 = {
                'CLEANED CONTENT': [
                    'Key in your data in this column',
                    'And extend the index column (left of this column) to the length of your data',
                    'You may safely delete the preloaded data in this column'
                ]
            }
            df_template_0 = pd.DataFrame(data=frame0)
            df_template_1 = pd.DataFrame(data=frame1)

            template = st.selectbox('Choose Template', ('General', 'DTM'))
            if template == 'General':
                st.markdown('Download template from [downloads/template.csv](downloads/template.csv)')
                df_template_0.to_csv(str(DOWNLOAD_PATH / 'template.csv'))
            elif template == 'DTM':
                st.markdown('Download template from [downloads/template_dtm.csv](downloads/template_dtm.csv)')
                df_template_1.to_csv(str(DOWNLOAD_PATH / 'template_dtm.csv'))

            st.markdown('## Project Contributors: \n\n'
                        '* Ong Jung Yi, PNSD\n\n'
                        '* George Tay, PNSD\n\n')
            st.markdown('## App Modules\n'
                        'Select the following available modules implemented:')

        # PAGE SELECTOR
        page = st.sidebar.selectbox('Data Science and NLP Functions',
                                    self.pages,
                                    format_func=lambda page: page['title'])
        # RUN THE APP
        page['function']()
