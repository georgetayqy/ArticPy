""""""
from fastapi import FastAPI
from endpoints.lca import clean, modify, query
from endpoints.dtm import dtm
from endpoints.mt import model_trainer
from endpoints.tk import toolkit_nlp

# instantiate the app
app = FastAPI()

# add the routers
app.include_router(clean.router)
app.include_router(modify.router)
app.include_router(query.router)
app.include_router(model_trainer.router)
app.include_router(dtm.router)
app.include_router(toolkit_nlp.router)


@app.get('/')
def root():
    return {'description': 'Welcome to the Homepage of the ArticPy API!'}


@app.get('/endpoints')
def info():
    return {
        'description': 'This page is the main directory where all endpoints for the API is branched out from.'
    }
