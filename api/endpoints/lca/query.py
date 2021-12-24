"""This file contains the code used for querying data from a given DataFrame that is passed to it in the JSON format"""

import pandas as pd
import fastapi

from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, HTTPException
from config import load_clean_visualise as lcv

router = APIRouter(prefix='/endpoints/lca/modify',
                   tags=['modify'],
                   responses={200: {'description': 'OK'},
                              404: {'description': 'Resource Not Found'},
                              415: {'description': 'Unsupported Media Type'}})


@router.post('/query')
def query(json_file, query_, data_column: str = '', match: bool = True):
    """
    Queries the input DataFrame in the form of JSON to find matching strings for query


    **json_file**:                   JSON data that is compatible with a pandas DataFrame

    **data_column**:                 The column name where the data to query is found

    **query_**:                      The string to query for in the data

    **match**:                       The strictness of query - True if query is case-sensitive
    """

    try:
        lcv['DATA'] = pd.read_csv(json_file, low_memory=False, encoding='latin1')
    except Exception as ex:
        raise HTTPException(status_code=415, detail=ex)
    else:
        if not lcv['DATA'].empty:
            try:
                # make a copy of the original dataframe to avoid mutating it with .loc
                temp = lcv['DATA'].copy()
                lcv['QUERY_DATA'] = temp.loc[temp[data_column].str.contains(query_, case=match)]
            except Exception as ex:
                return {'description': ex}
            else:
                data = {
                    'data': lcv['QUERY_DATA'].to_json()
                }
                return jsonable_encoder(data)
