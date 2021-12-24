"""This file contains code used to modify the input DataFrame in the JSON format"""

import pandas as pd
import fastapi
import pycountry

from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, HTTPException
from config import load_clean_visualise as lcv
from collections import Counter

router = APIRouter(prefix='/endpoints/lca/modify',
                   tags=['modify'],
                   responses={200: {'description': 'OK'},
                              404: {'description': 'Resource Not Found'},
                              415: {'description': 'Unsupported Media Type'}})


@router.post('/country-extraction')
def extract_country(json_file, data_column: str = ''):
    """
    Searches for instances of country names being mentioned in the DataFrame passed to it and returns the DataFrame
    modified with the country names extracted


    **json_file**:                   JSON Data that is compatible with a pandas DataFrame

    **data_column**:                 Column where the data of interest is found in
    """

    try:
        lcv['DATA'] = pd.read_json(json_file, low_memory=False, encoding='latin1')
    except Exception as ex:
        raise HTTPException(status_code=415, detail=ex)
    else:
        if not lcv['DATA'].empty:
            try:
                lcv['DATA'] = lcv['DATA'].astype(object)
                lcv['DATA']['COUNTRIES'] = lcv['DATA'][data_column].astype(str).apply(
                    lambda x: [country.name for country in pycountry.countries if country.name.lower() in x.lower()])
                new_list = lcv['DATA']['COUNTRIES'].to_list()
                temp = []
                for ls in new_list:
                    temp.extend(ls)
                zipped = list(zip(Counter(temp).keys(), Counter(temp).values()))

                lcv['GLOBE_DATA'] = pd.DataFrame(data=zipped, index=range(len(zipped)),
                                                 columns=['country', 'count'])
            except Exception as ex:
                return {'description': ex}
            else:
                data = {
                    'data': lcv['GLOBE_DATA'].to_json()
                }
                return jsonable_encoder(data)
