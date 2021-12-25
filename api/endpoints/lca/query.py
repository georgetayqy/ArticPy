"""This file contains the code used for querying data from a given DataFrame that is passed to it in the JSON format"""

import pandas as pd
import fastapi

from io import StringIO
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, HTTPException, File, UploadFile

router = APIRouter(prefix='/endpoints/lca/modify',
                   tags=['modify'],
                   responses={200: {'description': 'OK'},
                              404: {'description': 'Resource Not Found'},
                              415: {'description': 'Unsupported Media Type'}})


@router.post('/query')
async def query(file: UploadFile = File(...), ftype: str = 'csv', query_: str = None, data_column: str = '',
                match: bool = True):
    """
    Queries the input DataFrame in the form of JSON to find matching strings for query


    **file**: Data

    **ftype**: The file format to read the input data as

    **data_column**: The column name where the data to query is found

    **query_**: The string or list to query for in the data

    **match**: The strictness of query - True if query is case-sensitive
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
        if not raw_data.empty:
            try:
                temp = raw_data.copy()
                query_data = temp.loc[temp[data_column].str.contains(query_, case=match)]
            except Exception as ex:
                raise HTTPException(status_code=404, detail=ex)
            else:
                data = {
                    'data': query_data.to_json()
                }
                return jsonable_encoder(data)
        else:
            raise HTTPException(status_code=404, detail='Error: Data is not processed properly. Try again.')
