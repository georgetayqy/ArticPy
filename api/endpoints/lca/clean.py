import pathlib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
import texthero as hero
import fastapi

from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, HTTPException
from config import load_clean_visualise as lcv
from typing import Optional
from texthero import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def lemmatizeText(text):
    """
    This function iterates through the pandas dataframe and lemmatizes the words

    Parameters
    ----------
    :param text:                        Text to lemmatize (string)
    ----------
    """
    return [lemmatizer.lemmatize(word) for word in text]


# create an instance of the app
router = APIRouter(prefix='/endpoints/lca/clean',
                   tags=['clean'],
                   responses={200: {'description': 'OK'},
                              404: {'description': 'Resource Not Found'},
                              415: {'description': 'Unsupported Media Type'}})


@router.post('/no-clean')
def no_clean(json_file, data_column: str = 'data'):
    """
    This function takes in JSON data that is compatible with a pandas DataFrame, encodes it in the ASCII format and
    decodes it back into ASCII


    **json_file**:                      JSON Data

    **data_column**:                    Column in the pandas DataFrame to process
    """

    try:
        lcv['DATA'] = pd.read_json(json_file, low_memory=False, encoding='latin1')
    except Exception as ex:
        raise HTTPException(status_code=415, detail=ex)
    else:
        if not lcv['DATA'].empty:
            # en/decode the data first
            lcv['DATA'] = lcv['DATA'].astype(str)
            lcv['DATA'][data_column] = lcv['DATA'][data_column].str.encode('ascii', 'ignore') \
                .str.decode('ascii')
            lcv['DATA'] = pd.DataFrame(data=lcv['DATA'])
            lcv['DATA'] = lcv['DATA'].dropna()
            data = {
                'original': lcv['DATA'].to_json()
            }
            return jsonable_encoder(data)
        else:
            raise HTTPException(status_code=404, detail='Data is not properly loaded. Try again.')


@router.post('/simple-clean')
def simple_clean(json_file, data_column: str = 'data', tokenize: bool = True):
    """
    This function takes in JSON data that is compatible with a pandas DataFrame, encodes it in the ASCII format and
    decodes back into ASCII, and finally apply the 'Simple' Cleaning Pipeline and tokenzing the data (if the flag is
    set to True)


    **json_file**:                      JSON Data

    **data_column**:                    Column in the pandas DataFrame to process

    **tokenize**:                       Flag to determine whether to tokenize the data and to return it
    """

    try:
        lcv['DATA'] = pd.read_json(json_file, low_memory=False, encoding='latin1')
    except Exception as ex:
        raise HTTPException(status_code=415, detail=ex)
    else:
        if not lcv['DATA'].empty:
            # en/decode the data first
            lcv['DATA'] = lcv['DATA'].astype(str)
            lcv['DATA'][data_column] = lcv['DATA'][data_column].str.encode('ascii', 'ignore') \
                .str.decode('ascii')
            lcv['DATA'] = pd.DataFrame(data=lcv['DATA'])
            lcv['DATA'] = lcv['DATA'].dropna()

            # now we clean it
            try:
                lcv['CLEANED_DATA'] = lcv['DATA'][[data_column]]
                lcv['CLEANED_DATA']['CLEANED CONTENT'] = hero.clean(lcv['CLEANED_DATA'][data_column],
                                                                    lcv['SIMPLE_PIPELINE'])
                lcv['CLEANED_DATA']['CLEANED CONTENT'].replace('', np.nan, inplace=True)
                lcv['CLEANED_DATA'].dropna(inplace=True, subset=['CLEANED CONTENT'])

                lcv['CLEANED_DATA'] = lcv['CLEANED_DATA'].astype(str)
            except Exception as ex:
                return {'description': ex}
            else:
                if tokenize:
                    try:
                        lcv['CLEANED_DATA_TOKENIZE'] = hero.tokenize(lcv['CLEANED_DATA']['CLEANED CONTENT'])
                        lcv['CLEANED_DATA_TOKENIZE'] = lcv['CLEANED_DATA_TOKENIZE'].to_frame().astype(str)
                    except Exception as ex:
                        return {'description': ex}

            if not lcv['CLEANED_DATA'].empty and not lcv['CLEANED_DATA_TOKENIZE']:
                data = {
                    'cleaned_untokenized': lcv['CLEANED_DATA'].to_json(),
                    'cleaned_tokenized': lcv['CLEANED_DATA_TOKENIZE'].to_json()
                }
                return data
            elif not lcv['CLEANED_DATA'].empty and lcv['CLEANED_DATA_TOKENIZE'].empty:
                data = {
                    'cleaned_untokenized': lcv['CLEANED_DATA'].to_json()
                }
                return data
            elif lcv['CLEANED_DATA'].empty and not lcv['CLEANED_DATA_TOKENIZE'].empty:
                data = {
                    'cleaned_tokenized': lcv['CLEANED_DATA_TOKENIZE'].to_json()
                }
                return jsonable_encoder(data)
            else:
                raise HTTPException(status_code=404, detail='Data is not properly loaded. Try again.')


@router.post('/complex-clean')
def complex_clean(json_file, data_column: str = 'data', tokenize: bool = True, stopwords_list: str = None):
    """
    This function takes in JSON data that is compatible with a pandas DataFrame, encodes it in the ASCII format and
    decodes back into ASCII, and finally apply the 'Complex' Cleaning Pipeline and tokenzing the data (if the flag is
    set to True)


    **json_file**:                      JSON Data

    **data_column**:                    Column in the pandas DataFrame to process

    **tokenize**:                       Flag to determine whether to tokenize the data and to return it

    **stopwords_list**:                 A string (delimited by commas) or a list containing words to extend onto the
                                        default stopwords list.
    """

    try:
        lcv['DATA'] = pd.read_json(json_file, low_memory=False, encoding='latin1')
    except Exception as ex:
        raise HTTPException(status_code=415, detail=ex)
    else:
        # stopwords check
        if stopwords_list is not None:
            try:
                if len(stopwords_list) != 0:
                    lcv['STOPWORD_LIST'] = set(word.strip().lower() for word in stopwords_list.
                                               split(sep=','))
                    st.info(f'Stopwords accepted: {[word for word in lcv["STOPWORD_LIST"]]}!')
                    lcv['STOPWORD_LIST'] = stopwords.DEFAULT.union(stopwords_list)
                    lcv['FINALISE'] = True
            except Exception as ex:
                st.error(f'Error: {ex}')
        else:
            lcv['STOPWORD_LIST'] = stopwords.DEFAULT
            lcv['FINALISE'] = True

        # NO ELSE CONDITION AS ELSE CONDITION IS EXPLICITLY SPECIFIED IN THE PREVIOUS EXCEPTION/ERROR
        if lcv['FINALISE']:
            try:
                lcv['CLEANED_DATA'] = lcv['DATA'][[data_column]]

                # PREPROCESSING AND CLEANING
                lcv['CLEANED_DATA']['CLEANED CONTENT'] = hero.clean(lcv['CLEANED_DATA']
                                                                    [data_column],
                                                                    lcv['PIPELINE'])
                lcv['CLEANED_DATA']['CLEANED CONTENT'] = hero.remove_digits(lcv['CLEANED_DATA']
                                                                            ['CLEANED CONTENT'],
                                                                            only_blocks=False)
                lcv['CLEANED_DATA']['CLEANED CONTENT'] = hero.remove_stopwords(lcv['CLEANED_DATA']
                                                                               ['CLEANED CONTENT'],
                                                                               lcv["STOPWORD_LIST"])

                lcv['CLEANED_DATA_TOKENIZED'] = hero.tokenize(lcv['CLEANED_DATA']['CLEANED CONTENT'])
                lcv['CLEANED_DATA_TOKENIZED'] = lcv['CLEANED_DATA_TOKENIZED'].apply(lemmatizeText)

                # ACCEPT ONLY ENGLISH WORDS
                fin_list = [[word for word in text if word.lower() in lcv['ENGLISH_WORDS'] or not
                             word.isalpha()] for text in lcv['CLEANED_DATA_TOKENIZED']]

                # UPDATE TOKENS
                lcv['CLEANED_DATA']['CLEANED CONTENT'] = [' '.join(text) for text in fin_list]
                lcv['CLEANED_DATA_TOKENIZED'].update([str(text) for text in fin_list])
                lcv['CLEANED_DATA_TOKENIZED'] = lcv['CLEANED_DATA_TOKENIZED'].to_frame().astype(str)
                lcv['CLEANED_DATA']['CLEANED CONTENT'].replace('', np.nan, inplace=True)
                lcv['CLEANED_DATA'].dropna(subset=['CLEANED CONTENT'], inplace=True)
                lcv['CLEANED_DATA'] = lcv['CLEANED_DATA'].astype(str)
            except Exception as ex:
                return {'description': ex}

            if not lcv['CLEANED_DATA'].empty and not lcv['CLEANED_DATA_TOKENIZED'].empty:
                if tokenize:
                    data = {
                        'original': lcv['DATA'].to_json(),
                        'cleaned_untokenized': lcv['CLEANED_DATA'].to_json(),
                        'cleaned_tokenized': lcv['CLEANED_DATA_TOKENIZED'].to_json()
                    }
                    return jsonable_encoder(data)
                else:
                    data = {
                        'original': lcv['DATA'].to_json(),
                        'cleaned_tokenized': lcv['CLEANED_DATA'].to_json()
                    }
                    return jsonable_encoder(data)
            elif not lcv['CLEANED_DATA'].empty and lcv['CLEANED_DATA_TOKENIZED'].empty:
                data = {
                    'original': lcv['DATA'].to_json(),
                    'cleaned_untokenized': lcv['CLEANED_DATA'].to_json()
                }
                return jsonable_encoder(data)
            elif lcv['CLEANED_DATA'].empty and not lcv['CLEANED_DATA_TOKENIZED'].empty:
                if tokenize:
                    data = {
                        'original': lcv['DATA'].to_json(),
                        'cleaned_tokenized': lcv['CLEANED_DATA_TOKENIZED'].to_json()
                    }
                    return jsonable_encoder(data)
                else:
                    data = {
                        'original': lcv['DATA'].to_json()
                    }
                    return jsonable_encoder(data)
            elif lcv['CLEANED_DATA'].empty and lcv['CLEANED_DATA_TOKENIZED'].empty:
                raise HTTPException(status_code=404, detail='Data is not properly loaded. Try again.')
