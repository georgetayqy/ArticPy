"""
Generates a helper class to assist with the generation of multiple Streamlit apps

This file also allows users to set the app config for the app
"""

# IMPORT STREAMLIT
import streamlit as st


# DEFINE THE MULTIPAGE CLASS TO MANAGE THE APPS
class MultiPage:
    """
    Combines and manages the different modules within the streamlit application
    """

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
            st.markdown('# ArticPy ❄\n'
                        'An app built to simplify and condense NLP tasks into one simple yet powerful Interface.')

            # DEFINE THE TITLE AND CONTENTS OF THE MAIN PAGE
            st.markdown('## App Modules\n'
                        'Select the following available modules:')

        # PAGE SELECTOR
        page = st.sidebar.selectbox('NLP Functions',
                                    self.pages,
                                    format_func=lambda page: page['title'])
        # RUN THE APP
        try:
            page['function']()
        except ValueError:
            page['function']()
