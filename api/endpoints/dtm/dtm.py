import pandas as pd

from typing import Union
from io import StringIO
from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.encoders import jsonable_encoder
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

router = APIRouter(prefix='/endpoints',
                   tags=['dtm'],
                   responses={200: {'description': 'OK'},
                              404: {'description': 'Resource Not Found'},
                              415: {'description': 'Unsupported Media Type'}})


@router.post('/dtm')
async def dtm(file: UploadFile = File(...), ftype: str = 'csv', data_column: str = 'data') -> dict:
    """
    This function takes in CSV data that is compatible with a pandas DataFrame, creates a Document-Term Matrix and
    returns it to the user in JSON format


    **file**: Data

    **ftype**: The file format to read the input data as

    **data_column**: Column in the pandas DataFrame to process
    """

    try:
        if ftype == 'csv':
            raw_data = pd.read_csv(StringIO(str(file.file.read(), 'latin1')), encoding='latin1').astype(str)
        elif ftype == 'xlsx':
            raw_data = pd.read_excel(StringIO(str(file.file.read(), 'utf-8')), engine='openpyxl').astype(str)
        elif ftype == 'json':
            raw_data = pd.read_json(StringIO(str(file.file.read(), 'utf-8'))).astype(str)
        else:
            raise HTTPException(status_code=415, detail='Error: File format input is not supported. Try again.')
    except Exception as ex:
        raise HTTPException(status_code=415, detail=ex)
    else:
        counter_object = CountVectorizer(stop_words=stopwords.words('english'))
        word_string = ' '.join(raw_data[data_column])

        dict_data = {
            'text': word_string
        }

        series_data = pd.DataFrame(data=dict_data, index=[0])
        series_data = counter_object.fit_transform(series_data.text)
        dtm_ = pd.DataFrame(series_data.toarray(),
                            columns=counter_object.get_feature_names(),
                            index=[0])

        if not dtm_.empty:
            dtm_copy = dtm_.copy().transpose()
            dtm_copy.columns = ['Word Frequency']
            dtm_copy.sort_values(by=['Word Frequency'], ascending=False, inplace=True)
            data = {
                'dtm': dtm_copy.to_json()
            }
            return jsonable_encoder(data)
        else:
            raise HTTPException(status_code=404, detail='Error: Document-Term Matrix was not properly prepared. Try '
                                                        'again.')
