import pathlib
import nltk
import numpy as np
import pandas as pd
import texthero as hero
import fastapi

from texthero import preprocessing
from io import StringIO
from typing import Union
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, HTTPException, File, UploadFile
from texthero import stopwords
from nltk.stem import WordNetLemmatizer

# define constants
lemmatizer = WordNetLemmatizer()
SIMPLE_PIPELINE = [
    preprocessing.remove_html_tags,
    preprocessing.remove_diacritics,
    preprocessing.remove_whitespace,
    preprocessing.remove_urls,
    preprocessing.drop_no_content
    ]
PIPELINE = [
    preprocessing.fillna,
    preprocessing.lowercase,
    preprocessing.remove_punctuation,
    preprocessing.remove_html_tags,
    preprocessing.remove_diacritics,
    preprocessing.remove_whitespace,
    preprocessing.remove_urls,
    preprocessing.drop_no_content
    ]


def lemmatizeText(text):
    """
    This function iterates through the pandas dataframe and lemmatizes the words

    Parameters
    ----------
    :param text:                        Text to lemmatize (string)
    ----------
    """
    return [lemmatizer.lemmatize(word) for word in text]


# API router
router = APIRouter(prefix='/endpoints/lca/clean',
                   tags=['clean'],
                   responses={200: {'description': 'OK'},
                              404: {'description': 'Resource Not Found'},
                              415: {'description': 'Unsupported Media Type'}})


@router.post('/no-clean')
async def no_clean(file: UploadFile = File(...), ftype: str = 'csv', data_column: str = 'data'):
    """
    This function takes in JSON data that is compatible with a pandas DataFrame, encodes it in the ASCII format and
    decodes it back into ASCII


    **file**: Data

    **ftype**: The file format to read the input data as

    **data_column**:                    Column in the pandas DataFrame to process
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
        raise HTTPException(status_code=404, detail=ex)
    else:
        if not raw_data.empty:
            raw_data[data_column] = raw_data[data_column].str.encode('ascii', 'ignore') \
                .str.decode('ascii')
            raw_data = pd.DataFrame(data=raw_data)
            raw_data = raw_data.dropna()
            data = {
                'original': raw_data.to_json()
            }
            return jsonable_encoder(data)
        else:
            raise HTTPException(status_code=404, detail='Data is not properly loaded. Try again.')


@router.post('/simple-clean')
async def simple_clean(file: UploadFile = File(...), ftype: str = 'csv', data_column: str = 'data', tokenize: bool = True):
    """
    This function takes in JSON data that is compatible with a pandas DataFrame, encodes it in the ASCII format and
    decodes back into ASCII, and finally apply the 'Simple' Cleaning Pipeline and tokenizing the data (if the flag is
    set to True)


    **file**: Data

    **ftype**: The file format to read the input data as

    **data_column**: Column in the pandas DataFrame to process

    **tokenize**: Flag to determine whether to tokenize the data and to return it
    """

    cleaned_data_tokenized = None

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
            raw_data[data_column] = raw_data[data_column].str.encode('ascii', 'ignore') \
                .str.decode('ascii')
            raw_data = pd.DataFrame(data=raw_data)
            raw_data = raw_data.dropna()

            try:
                cleaned_data = raw_data[[data_column]]
                cleaned_data['CLEANED CONTENT'] = hero.clean(cleaned_data[data_column], SIMPLE_PIPELINE)
                cleaned_data['CLEANED CONTENT'].replace('', np.nan, inplace=True)
                cleaned_data.dropna(inplace=True, subset=['CLEANED CONTENT'])

                cleaned_data = cleaned_data.astype(str)
            except Exception as ex:
                raise HTTPException(status_code=404, detail=ex)
            else:
                if tokenize:
                    try:
                        cleaned_data_tokenized = hero.tokenize(cleaned_data['CLEANED CONTENT']).to_frame().astype(str)
                    except Exception as ex:
                        raise HTTPException(status_code=404, detail=ex)

            if not cleaned_data.empty and not cleaned_data_tokenized.empty:
                data = {
                    'cleaned_untokenized': cleaned_data.to_json(),
                    'cleaned_tokenized': cleaned_data_tokenized.to_json()
                }
                return data
            elif not cleaned_data.empty and cleaned_data_tokenized.empty:
                data = {
                    'cleaned_untokenized': cleaned_data.to_json()
                }
                return data
            elif cleaned_data.empty and not cleaned_data_tokenized.empty:
                data = {
                    'cleaned_tokenized': cleaned_data_tokenized.to_json()
                }
                return jsonable_encoder(data)
            else:
                raise HTTPException(status_code=404, detail='Data is not properly loaded. Try again.')


@router.post('/complex-clean')
async def complex_clean(file: UploadFile = File(...), ftype: str = 'csv', data_column: str = 'data',
                        tokenize: bool = True, stopwords_list: Union[str, list] = None):
    """
    This function takes in JSON data that is compatible with a pandas DataFrame, encodes it in the ASCII format and
    decodes back into ASCII, and finally apply the 'Complex' Cleaning Pipeline and tokenzing the data (if the flag is
    set to True)


    **file**: Data

    **ftype**: The file format to read the input data as

    **data_column**: Column in the pandas DataFrame to process

    **tokenize**: Flag to determine whether to tokenize the data and to return it

    **stopwords_list**: A string (delimited by commas) or a list containing words to extend onto the default stopwords
                        list.
    """

    finalised = None

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
        # stopwords check
        if stopwords_list is not None:
            if type(stopwords_list) is str:
                try:
                    if len(stopwords_list) != 0:
                        stopwords_list = stopwords.DEFAULT.union(set(word.strip().lower() for word in
                                                                     stopwords_list.split(sep=',')))
                        finalised = True
                except Exception as ex:
                    raise HTTPException(status_code=404, detail=ex)
            elif type(stopwords_list) is list:
                stopwords_list = stopwords.DEFAULT.union(stopwords_list)
                finalised = True
            else:
                raise HTTPException(status_code=404, detail='Invalid type for stopwords_list ')
        else:
            stopwords_list = stopwords.DEFAULT
            finalised = True

        if finalised:
            try:
                cleaned_data = raw_data[[data_column]]
                cleaned_data['CLEANED CONTENT'] = hero.clean(cleaned_data[data_column], PIPELINE)
                cleaned_data['CLEANED CONTENT'] = hero.remove_digits(cleaned_data['CLEANED CONTENT'], only_blocks=False)
                cleaned_data['CLEANED CONTENT'] = hero.remove_stopwords(cleaned_data['CLEANED CONTENT'], stopwords_list)
                cleaned_data_tokenized = hero.tokenize(cleaned_data['CLEANED CONTENT'])
                cleaned_data_tokenized = cleaned_data_tokenized.apply(lemmatizeText)

                fin_list = [[word for word in text if word.lower() in set(nltk.corpus.words.words()) or not
                             word.isalpha()] for text in cleaned_data_tokenized]

                cleaned_data['CLEANED CONTENT'] = [' '.join(text) for text in fin_list]
                cleaned_data_tokenized.update([str(text) for text in fin_list])
                cleaned_data_tokenized = cleaned_data_tokenized.to_frame().astype(str)
                cleaned_data['CLEANED CONTENT'].replace('', np.nan, inplace=True)
                cleaned_data.dropna(subset=['CLEANED CONTENT'], inplace=True)
                cleaned_data = cleaned_data.astype(str)
            except Exception as ex:
                raise HTTPException(status_code=404, detail=ex)
            else:
                if not cleaned_data.empty and not cleaned_data_tokenized.empty:
                    if tokenize:
                        data = {
                            'original': raw_data.to_json(),
                            'cleaned_untokenized': cleaned_data.to_json(),
                            'cleaned_tokenized': cleaned_data_tokenized.to_json()
                        }
                        return jsonable_encoder(data)
                    else:
                        data = {
                            'original': raw_data.to_json(),
                            'cleaned_tokenized': cleaned_data.to_json()
                        }
                        return jsonable_encoder(data)
                elif not cleaned_data.empty and cleaned_data_tokenized.empty:
                    data = {
                        'original': raw_data.to_json(),
                        'cleaned_untokenized': cleaned_data.to_json()
                    }
                    return jsonable_encoder(data)
                elif cleaned_data.empty and not cleaned_data_tokenized.empty:
                    if tokenize:
                        data = {
                            'original': raw_data.to_json(),
                            'cleaned_tokenized': cleaned_data_tokenized.to_json()
                        }
                        return jsonable_encoder(data)
                    else:
                        data = {
                            'original': raw_data.to_json()
                        }
                        return jsonable_encoder(data)
                elif cleaned_data.empty and cleaned_data_tokenized.empty:
                    raise HTTPException(status_code=404, detail='Data is not properly loaded. Try again.')
        else:
            raise HTTPException(status_code=404, detail='Data is not properly processed. Try again.')