from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from config import dtm
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


router = APIRouter(prefix='/endpoints',
                   tags=['dtm'],
                   responses={200: {'description': 'OK'},
                              404: {'description': 'Resource Not Found'},
                              415: {'description': 'Unsupported Media Type'}})


@router.post('/dtm')
def dtm(json_file, data_column: str = 'data') -> str:
    """
    This function takes in JSON data that is compatible with a pandas DataFrame, creates a Document-Term Matrix and
    returns it to the user in JSON format


    **json_file**:                      JSON Data

    **data_column**:                    Column in the pandas DataFrame to process
    """

    try:
        dtm['DATA'] = pd.read_json(json_file, low_memory=False, encoding='latin1')
    except Exception as ex:
        raise HTTPException(status_code=415, detail=ex)
    else:
        dtm['DATA'] = dtm['DATA'].astype(str)
        counter_object = CountVectorizer(stop_words=stopwords.words('english'))

        # CONCATENATE ALL STR VALUES INTO LIST: OPTIMISED
        word_string = ' '.join(dtm['DATA'][data_column])

        # CREATE A NEW DF TO PARSE
        dict_data = {
            'text': word_string
        }
        series_data = pd.DataFrame(data=dict_data, index=[0])

        # FIT-TRANSFORM THE DATA
        series_data = counter_object.fit_transform(series_data.text)

        # CONVERT THE FITTED DATA INTO A PANDAS DATAFRAME TO USE FOR THE DTMs
        dtm['DTM'] = pd.DataFrame(series_data.toarray(),
                                  columns=counter_object.get_feature_names(),
                                  index=[0])
        if not dtm['DTM'].empty:
            dtm['DTM_copy'] = dtm['DTM'].copy().transpose()
            dtm['DTM_copy'].columns = ['Word Frequency']
            dtm['DTM_copy'].sort_values(by=['Word Frequency'], ascending=False, inplace=True)
            data = {
                'dtm': dtm['DTM_copy'].to_json()
            }
            return jsonable_encoder(data)
        else:
            raise HTTPException(status_code=404, detail='Error: Document-Term Matrix was not properly prepared. Try '
                                                        'again.')
