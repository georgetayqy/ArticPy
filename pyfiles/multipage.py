"""
Generates a helper class to assist with the generation of multiple Streamlit apps

This file also allows users to set the app config for the app
"""

# IMPORT STREAMLIT
import streamlit as st
from pyfiles.pages import load_clean_visualise, document_term_matrix, toolkit_nlp


# DEFINE THE MULTIPAGE CLASS TO MANAGE THE APPS
class MultiPage:
    """
    Combines and manages the different modules within the streamlit application
    """

    def __init__(self) -> None:
        """Constructor to generate a list which will store all our applications as an instance variable."""

        # SAVE FUNCTIONS TO SESSION STATE TO PRESERVE FUNCTIONS ACROSS RERUNS
        if 'pages' not in st.session_state:
            st.session_state.pages = [{'title': 'Load, Clean and Visualise', 'function': load_clean_visualise.app},
                                      {'title': 'Document-Term Matrix', 'function': document_term_matrix.app},
                                      {'title': 'NLP Toolkit', 'function': toolkit_nlp.app}]

    def add_page(self, title, func) -> None:
        """
        Class Method to add pages to the app

        Arguments
        ----------
        title:                  The title of page which we are adding to the list of apps
        func:                   Python function to render this page in Streamlit
        ----------
        """

        st.session_state.pages.append({
            "title": title,
            "function": func
        })

    def run(self):
        """
        Dropdown menu to select the page to run
        """

        with st.sidebar.container():
            st.markdown('# ArticPy ❄\n'
                        'An app built to simplify and condense NLP tasks into one simple yet powerful Interface.')

            # DEFINE THE TITLE AND CONTENTS OF THE MAIN PAGE
            st.markdown('## App Modules\n'
                        'Select the following available modules:')

        # PAGE SELECTOR
        page = st.sidebar.selectbox('NLP Functions',
                                    st.session_state.pages,
                                    format_func=lambda page: page['title'])
        # RUN THE APP
        try:
            page['function']()
        except ValueError:
            page['function']()
