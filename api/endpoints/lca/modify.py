"""This file contains code used to modify the input DataFrame in the JSON format"""

import pandas as pd
import fastapi
import pycountry

from io import StringIO
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, HTTPException, File, UploadFile
from collections import Counter

router = APIRouter(prefix='/endpoints/lca/modify',
                   tags=['modify'],
                   responses={200: {'description': 'OK'},
                              404: {'description': 'Resource Not Found'},
                              415: {'description': 'Unsupported Media Type'}})


@router.post('/country-extraction')
async def extract_country(file: UploadFile = File(...), ftype: str = 'csv', data_column: str = ''):
    """
    Searches for instances of country names being mentioned in the DataFrame passed to it and returns the DataFrame
    modified with the country names extracted


    **file**: Data

    **ftype**: The file format to read the input data as

    **data_column**:                 Column where the data of interest is found in
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
                raw_data = raw_data.astype(object)
                raw_data['COUNTRIES'] = raw_data[data_column].astype(str).apply(
                    lambda x: [country.name for country in pycountry.countries if country.name.lower() in x.lower()])
                new_list = raw_data['COUNTRIES'].to_list()
                temp = []
                for ls in new_list:
                    temp.extend(ls)
                zipped = list(zip(Counter(temp).keys(), Counter(temp).values()))

                globe_data = pd.DataFrame(data=zipped, index=range(len(zipped)), columns=['country', 'count'])
            except Exception as ex:
                raise HTTPException(status_code=404, detail=ex)
            else:
                data = {
                    'data': globe_data.to_json()
                }
                return jsonable_encoder(data)
        else:
            raise HTTPException(status_code=404, detail='Error: Data is not processed properly. Try again.')
