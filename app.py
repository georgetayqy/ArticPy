# CUSTOM PAGE IMPORTS
from pyfiles.multipage import MultiPage
from pyfiles.pages import load_clean_visualise, document_term_matrix, toolkit_nlp, toolkit_discriminator

# INSTANTIATE THE APP
app = MultiPage()

# DEFINE THE PAGES AND THE APPS THEY CONTAIN
app.add_page('Load, Clean and Visualise Data', load_clean_visualise.app)
app.add_page('DTM and Word Frequency Analysis', document_term_matrix.app)
app.add_page('NLP Toolkit', toolkit_nlp.app)


# RUN THE APP
app.run()
