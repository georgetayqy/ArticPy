# INIT STREAMLIT CONFIG
import streamlit as st
st.set_page_config(page_title='ArticPy',
                   page_icon='❄',
                   menu_items={
                       'Report a bug': 'https://github.com/asdfghjkxd/ArticPy/issues',
                       'About': '## ArticPy ❄ \n'
                                'An app built to simplify and condense NLP tasks into one simple yet powerful '
                                'Interface.\n\n'
                                '### Project Contributors: \n\n'
                                'Ong Jung Yi, PNSD & George Tay, PNSD'
                   })

# CUSTOM PAGE IMPORTS
from pyfiles.multipage import MultiPage
from pyfiles.pages import load_clean_visualise #, document_term_matrix, toolkit_nlp, model_trainer

# INSTANTIATE THE APP
app = MultiPage()

# DEFINE THE PAGES AND THE APPS THEY CONTAIN
app.add_page('Load, Clean and Visualise Data', load_clean_visualise.app)
# app.add_page('DTM and Word Frequency Analysis', document_term_matrix.app)
# app.add_page('NLP Toolkit', toolkit_nlp.app)
# app.add_page('NLP Model Trainer', model_trainer.app)

# RUN THE APP
try:
    app.run()
except ValueError:
    app.run()
