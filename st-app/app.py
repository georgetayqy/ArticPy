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

# INSTANTIATE THE APP
app = MultiPage()

# RUN THE APP
try:
    app.run()
except ValueError:
    app.run()
