import logging
import os
from collections import Counter
from heapq import nlargest
from string import punctuation
import numpy as np
import pandas as pd
import spacy
import streamlit as st
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
import nltk
import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.sklearn
import textattack.models.wrappers
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
import transformers

from io import StringIO
from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.encoders import jsonable_encoder
from operator import itemgetter
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, AutoModelForSequenceClassification
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from spacy import displacy
from wordcloud import WordCloud
from textblob import TextBlob

# API router
router = APIRouter(prefix='/endpoints/toolkit',
                   tags=['toolkit'],
                   responses={200: {'description': 'OK'},
                              404: {'description': 'Resource Not Found'},
                              415: {'description': 'Unsupported Media Type'}})

# file counter
fc = 0


def summarise(text, stopwords, pos_tag, nlp, sent_count):
    """
    This function summarise the text dataframe

    Parameters
    ----------
    text:                               DataFrame
    nlp:                                NLP model
    pos_tag:                            Text pos tag
    stopwords:                          Stopwords
    sent_count:                         Number of sentences to summarise to
    ----------
    """

    try:
        # DEFINE LISTS AND DICTS
        keyword = []
        sent_strength = {}
        data = nlp(str(text))

        # EXTRACT KEYWORDS FROM TEXT
        for token in data:
            if token.text in stopwords or token.text in punctuation:
                continue
            if token.pos_ in pos_tag:
                keyword.append(token.text)

        # COUNT THE FREQUENCY OF WORDS
        freq_word = Counter(keyword)
        max_freq = Counter(keyword).most_common(1)[0][1]
        for word in freq_word.keys():
            freq_word[word] = (freq_word[word] / max_freq)

        # CALCULATE SENTENCE SCORES
        for sent in data.sents:
            for word in sent:
                if word.text in freq_word.keys():
                    if sent in sent_strength.keys():
                        sent_strength[sent] += freq_word[word.text]
                    else:
                        sent_strength[sent] = freq_word[word.text]

        # CONCATENATE THE STRINGS IN THE LIST TO A LARGER STRING
        summarized_sentences = nlargest(sent_count, sent_strength, key=sent_strength.get)
        final_sentences = [w.text for w in summarized_sentences]
        summary = ' '.join(final_sentences)
    except Exception:
        return text
    else:
        return summary


def modelIterator(model, vectoriser, top_n, vb=True):
    """
    This function prints out and returns the extracted topics for the NLP model passed on to it

    Parameters
    ----------
    model:                              NLP Model
    vectoriser:                         Vectorised text
    top_n:                              Number of Topics to return
    vb:                                 Verbose tag (will print out the topics if set to True
    ---------
    """
    frame_list = []

    for id_, topic in enumerate(model.components_):
        lister = [(vectoriser.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]]
        df = pd.DataFrame(data=lister,
                          index=range(len(lister)),
                          columns=['word', 'weight'])

        if vb:
            st.markdown(f'### Topic {id_}')
            st.dataframe(df)

        frame_list.append(df)

    return frame_list


def dominantTopic(vect, model, n_words):
    """
    Returns the topic text

    Parameters
    ----------
    vect:                               Vectorizer used
    model:                              NLP Model
    n_words:                            Number of Topics to return
    ----------
    """
    kw = np.array(vect.get_feature_names())
    topic_kw = []
    for weights in model.components_:
        top_kw = (-weights).argsort()[:n_words]
        topic_kw.append(kw.take(top_kw))

    return topic_kw


@router.post('/wordcloud')
async def wordcloud(file: UploadFile = File(...), ftype: str = 'csv', data_column: str = 'data', max_word: int = 200,
                    contour: int = 3, width: int = 800, height: int = 400, colour: str = 'steelblue'):
    """
    Wordcloud creation
    
    
    **file**: Data

    **ftype**: The file format to read the input data as

    **data_column**: Column in the pandas DataFrame to process
    
    **max_word**: Max number of words to render in the wordcloud image
    
    **contour**: Contour width
    
    **width**: Width of the wordcloud image
    
    **height**: Height of the wordcloud image
    
    **colour**: Colour of the background
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
            raw_data = raw_data[[data_column]]
            wc = WordCloud(background_color='white',
                           max_words=max_word,
                           contour_width=contour,
                           width=width,
                           height=height,
                           contour_color=colour)
            wc.generate(' '.join(raw_data[data_column]))
            img = wc.to_image()
            data = {
                'image': str(img.tobytes())
            }

            return jsonable_encoder(data)
        else:
            raise HTTPException(status_code=404, detail='Error: Document-Term Matrix was not properly prepared. Try '
                                                        'again.')


@router.post('/ner')
async def ner(file: UploadFile = File(...), ftype: str = 'csv', data_column: str = 'data',
              model: str = 'en_core_web_sm', one_datapoint_analyser: int = None):
    """
    Conduct NER analysis
    
    
    **file**: Data

    **ftype**: The file format to read the input data as

    **data_column**: Column in the pandas DataFrame to process
    
    **model**: spaCy model to load
    
    **one_datapoint_analyser**: The datapoint to render into HTML format
    """

    NLP = None

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
            # init the required columns
            raw_data = raw_data[[data_column]]
            raw_data['NER'] = ''
            raw_data['COMPILED_LABELS'] = ''
            raw_data = raw_data.astype(str)

            if model == 'en_core_web_sm':
                try:
                    NLP = spacy.load('en_core_web_sm')
                except OSError:
                    logging.warning('Model not found, downloading...')
                    try:
                        os.system('python -m spacy download en_core_web_sm')
                    except Exception as ex:
                        logging.error(f'Unable to download Model. Error: {ex}')
                        raise HTTPException(status_code=415, detail=ex)
                except Exception as ex:
                    raise HTTPException(status_code=415, detail=ex)
            elif model == 'en_core_web_lg':
                try:
                    NLP = spacy.load('en_core_web_lg')
                except OSError:
                    logging.warning('Model not found, downloading...')
                    try:
                        os.system('python -m spacy download en_core_web_lg')
                    except Exception as ex:
                        logging.error(f'Unable to download Model. Error: {ex}')
                        raise HTTPException(status_code=415, detail=ex)
                except Exception as ex:
                    raise HTTPException(status_code=415, detail=ex)

            for index in range(len(raw_data)):
                temp_nlp = NLP(raw_data[data_column][index])
                raw_data.at[index, 'NER'] = str(list(zip([word.text for word in temp_nlp.ents],
                                                         [word.label_ for word in temp_nlp.ents])))
                raw_data.at[index, 'COMPILED_LABELS'] = str(list(set([word.label_ for word in temp_nlp.ents])))

            if one_datapoint_analyser is not None:
                cpy = raw_data.copy()
                temp = cpy[data_column][one_datapoint_analyser]
                render = displacy.render(list(NLP(str(temp)).sents),
                                         style='ent',
                                         page=True)
                data = {
                    'data': raw_data.to_json(),
                    'render': render
                }
                return jsonable_encoder(data)
            else:
                data = {
                    'data': raw_data.to_json()
                }
                return jsonable_encoder(data)
        else:
            raise HTTPException(status_code=404, detail='Error: Data not loaded properly. Try again.')


@router.post('/pos')
async def pos(file: UploadFile = File(...), ftype: str = 'csv', data_column: str = '',
              model: str = 'en_core_web_sm', one_datapoint_analyser: int = None, compact: bool = True,
              colour: str = 'steelblue', bg: str = 'white'):
    """
    Conduct POS tagging
    
    
    **file**: Data

    **ftype**: The file format to read the input data as

    **data_column**: Column in the pandas DataFrame to process
    
    **model**: spaCy model to load
    
    **one_datapoint_analyser**: The datapoint to render into HTML format
    
    **compact**: Compact the renders
    
    **colour**: Colour of the words in the render
    
    **bg**: Colour of the background
    """

    NLP = None

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
            raw_data = raw_data[[data_column]]
            raw_data['POS'] = ''
            raw_data = raw_data.astype(str)

            if model == 'en_core_web_sm':
                try:
                    NLP = spacy.load('en_core_web_sm')
                except OSError:
                    logging.warning('Model not found, downloading...')
                    try:
                        os.system('python -m spacy download en_core_web_sm')
                    except Exception as ex:
                        logging.error(f'Unable to download Model. Error: {ex}')
                        raise HTTPException(status_code=415, detail=ex)
                except Exception as ex:
                    raise HTTPException(status_code=415, detail=ex)
            elif model == 'en_core_web_lg':
                try:
                    NLP = spacy.load('en_core_web_lg')
                except OSError:
                    logging.warning('Model not found, downloading...')
                    try:
                        os.system('python -m spacy download en_core_web_lg')
                    except Exception as ex:
                        logging.error(f'Unable to download Model. Error: {ex}')
                        raise HTTPException(status_code=415, detail=ex)
                except Exception as ex:
                    raise HTTPException(status_code=415, detail=ex)

            for index in range(len(raw_data)):
                temp_nlp = NLP(raw_data[data_column][index])
                raw_data.at[index, 'POS'] = str(list(zip([str(word) for word in temp_nlp],
                                                         [word.pos_ for word in temp_nlp])))
                raw_data.at[index, 'COMPILED_LABELS'] = str(list(set([word.pos_ for word in temp_nlp])))

            if one_datapoint_analyser is not None:
                cpy = raw_data.copy()
                temp = cpy[data_column][one_datapoint_analyser]
                render = displacy.render(list(NLP(str(temp)).sents),
                                         style='dep',
                                         options={
                                             'compact': compact,
                                             'color': colour,
                                             'bg': bg,
                                         })
                data = {
                    'data': raw_data.to_json(),
                    'render': render
                }
                return jsonable_encoder(data)
            else:
                data = {
                    'data': raw_data.to_json()
                }
                return jsonable_encoder(data)
        else:
            raise HTTPException(status_code=415, detail='Error: Data not loaded properly. Try again.')


@router.post('/summarise')
def summarise(file: UploadFile = File(...), ftype: str = 'csv', data_column: str = 'data', mode: str = 'basic',
              model: str = 'en_core_web_sm', sentence_len: int = 3, min_words: int = 80, max_words: str = 150,
              max_tensor: int = 512):
    """
    Summarise texts
    
    
    **file**: Data

    **ftype**: The file format to read the input data as

    **data_column**: Column in the pandas DataFrame to process
    
    **mode**: Define whether or not to conduct 'basic' or 'advanced' summarisation on input data
    
    **model**: spaCy model to load
    
    **sentence_len**: The maximum length of sentence to return
    
    **min_words**: The minimum number of words to include in the summary
    
    **max_words**: The maximum number of words to include in the summary
    
    **max_tensor**: The maximum number of input tensors for advanced summarisation process
    """

    NLP = None

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
        # load up the data first
        raw_data = raw_data[[data_column]]
        raw_data['SUMMARY'] = np.nan
        raw_data = raw_data.astype(str)

        if not raw_data.empty:
            if mode == 'basic':
                if model == 'en_core_web_sm':
                    try:
                        NLP = spacy.load('en_core_web_sm')
                    except OSError:
                        logging.warning('Model not found, downloading...')
                        try:
                            os.system('python -m spacy download en_core_web_sm')
                        except Exception as ex:
                            logging.error(f'Unable to download Model. Error: {ex}')
                            raise HTTPException(status_code=415, detail=ex)
                    except Exception as ex:
                        raise HTTPException(status_code=415, detail=ex)
                elif model == 'en_core_web_lg':
                    try:
                        NLP = spacy.load('en_core_web_lg')
                    except OSError:
                        logging.warning('Model not found, downloading...')
                        try:
                            os.system('python -m spacy download en_core_web_lg')
                        except Exception as ex:
                            logging.error(f'Unable to download Model. Error: {ex}')
                            raise HTTPException(status_code=415, detail=ex)
                    except Exception as ex:
                        raise HTTPException(status_code=415, detail=ex)

                stopwords = list(STOP_WORDS)
                pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
                raw_data['SUMMARY'] = raw_data[data_column]. \
                    apply(lambda x: summarise(x, stopwords, pos_tag, NLP, sentence_len))
                data = {
                    'data': raw_data.to_json()
                }
                return jsonable_encoder(data)

            elif mode == 'advanced':
                if torch.cuda.is_available():
                    try:
                        torch.cuda.get_device_name(torch.cuda.current_device())
                    except AssertionError:
                        raise HTTPException(status_code=415, detail='Error: CUDA Device is not enabled. Try again.')
                    except Exception as ex:
                        raise HTTPException(status_code=415, detail=ex)
                    else:
                        tokenizer = AutoTokenizer.from_pretrained('t5-base')
                        model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)
                        raw_data = raw_data.astype(object)
                        raw_data['ENCODED'] = raw_data[data_column]. \
                            apply(lambda x: tokenizer.encode('summarize: ' + x,
                                                             return_tensors='pt',
                                                             max_length=max_tensor,
                                                             truncation=True))
                        raw_data['OUTPUTS'] = raw_data['ENCODED']. \
                            apply(lambda x: model.generate(x,
                                                           max_length=max_words,
                                                           min_length=min_words,
                                                           length_penalty=5.0,
                                                           num_beams=2))
                        raw_data['SUMMARISED'] = raw_data['OUTPUTS'].apply(
                            lambda x: tokenizer.decode(x[0]))
                        raw_data.drop(columns=['ENCODED', 'OUTPUTS'], inplace=True)
                        raw_data['SUMMARISED'] = raw_data['SUMMARISED']. \
                            str.replace('<pad> ', '').str.replace('</s>', '')
                        raw_data = raw_data.astype(str)
                        data = {
                            'data': raw_data.to_json()
                        }
                        return jsonable_encoder(data)

        else:
            raise HTTPException(status_code=415, detail='Error: Data not loaded properly. Try again.')


@router.post('/sentiment')
def sentiment(file: UploadFile = File(...), ftype: str = 'csv', data_column: str = 'data', model: str = 'vader',
              colour: str = '#2ACAEA'):
    """
    Conduct Sentiment Analysis
    
    
    **file**: Data

    **ftype**: The file format to read the input data as

    **data_column**: Column in the pandas DataFrame to process
    
    **model**: spaCy model to load
    
    **colour**: Colour of plots generated
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
            if model == 'vader':
                replacer = {
                    r"'": '',
                    r'[^\w\s]': ' ',
                    r' \d+': ' ',
                    r' +': ' '
                }

                raw_data['VADER SENTIMENT TEXT'] = raw_data[data_column]. \
                    replace(to_replace=replacer, regex=True)

                vader_analyser = SentimentIntensityAnalyzer()
                sent_score_list = []
                sent_label_list = []

                # scoring
                for i in raw_data['VADER SENTIMENT TEXT'].tolist():
                    sent_score = vader_analyser.polarity_scores(i)

                    if sent_score['compound'] > 0:
                        sent_score_list.append(sent_score['compound'])
                        sent_label_list.append('Positive')
                    elif sent_score['compound'] == 0:
                        sent_score_list.append(sent_score['compound'])
                        sent_label_list.append('Neutral')
                    elif sent_score['compound'] < 0:
                        sent_score_list.append(sent_score['compound'])
                        sent_label_list.append('Negative')

                raw_data['VADER OVERALL SENTIMENT'] = sent_label_list
                raw_data['VADER OVERALL SCORE'] = sent_score_list
                raw_data['VADER POSITIVE SCORING'] = [vader_analyser.polarity_scores(doc)['pos'] for doc in
                                                      raw_data['VADER SENTIMENT TEXT'].values.tolist()]
                raw_data['VADER NEUTRAL SCORING'] = [vader_analyser.polarity_scores(doc)['neu'] for doc in
                                                     raw_data['VADER SENTIMENT TEXT'].values.tolist()]
                raw_data['VADER NEGATIVE SCORING'] = [vader_analyser.polarity_scores(doc)['neg'] for doc in
                                                      raw_data['VADER SENTIMENT TEXT'].values.tolist()]

                # create plots
                hac_plot = ff.create_distplot([raw_data['VADER OVERALL SCORE'].tolist()],
                                              ['VADER'],
                                              colors=[colour],
                                              bin_size=0.25,
                                              curve_type='normal',
                                              show_rug=False,
                                              show_hist=False)
                hac_plot.update_layout(title_text='Distribution Plot',
                                       xaxis_title='VADER Score',
                                       yaxis_title='Frequency Density',
                                       legend_title='Frequency Density')
                data = {
                    'data': raw_data.to_json(),
                    'dot_image': hac_plot.to_image(format="png")

                }
                return jsonable_encoder(data)

            elif model == 'textblob':
                pol_list = []
                sub_list = []

                # scoring: polarity
                raw_data['POLARITY SCORE'] = raw_data[data_column]. \
                    apply(lambda x: TextBlob(x).sentiment.polarity)
                for i in raw_data['POLARITY SCORE'].tolist():
                    if float(i) > 0:
                        pol_list.append('Positive')
                    elif float(i) < 0:
                        pol_list.append('Negative')
                    elif float(i) == 0:
                        pol_list.append('Neutral')
                raw_data['POLARITY SENTIMENT'] = pol_list

                # scoring: subjectivity
                raw_data['SUBJECTIVITY SCORE'] = raw_data[data_column].apply(
                    lambda x: TextBlob(x).sentiment.subjectivity
                )
                for i in raw_data['SUBJECTIVITY SCORE'].tolist():
                    if float(i) < 0.5:
                        sub_list.append('Objective')
                    elif float(i) > 0.5:
                        sub_list.append('Subjective')
                    elif float(i) == 0.5:
                        sub_list.append('Neutral')
                raw_data['SUBJECTIVITY SENTIMENT'] = sub_list
                hac_plot = px.scatter(raw_data[['SUBJECTIVITY SCORE', 'POLARITY SCORE']],
                                      x='SUBJECTIVITY SCORE',
                                      y='POLARITY SCORE',
                                      labels={
                                          'SUBJECTIVITY SCORE': 'Subjectivity',
                                          'POLARITY SCORE': 'Polarity'
                                      })
                hac_plot1 = ff.create_distplot([raw_data['SUBJECTIVITY SCORE'].tolist(),
                                                raw_data['POLARITY SCORE'].tolist()],
                                               ['Subjectivity', 'Polarity'],
                                               curve_type='normal',
                                               show_rug=False,
                                               show_hist=False)
                data = {
                    'data': raw_data.to_json(),
                    'dot_image': hac_plot.to_image(format="png"),
                    'word_image': hac_plot1.to_image(format="png")
                }
                return jsonable_encoder(data)
        else:
            raise HTTPException(status_code=415, detail='Error: Data not loaded properly. Try again.')


@router.post('/modelling')
def topic_modelling(file: UploadFile = File(...), ftype: str = 'csv', data_column: str = 'data', model: str = 'lda',
                    num_topics: int = 10, max_features: int = 5000, max_iter: int = 10, min_df: int = 5,
                    max_df: float = 0.90, worker: int = 1, colour: str = 'steelblue', alpha: float = 0.10,
                    l1_ratio: float = 0.50):
    """
    Topic Modelling
    
    
    **file**: Data

    **ftype**: The file format to read the input data as

    **data_column**: Column in the pandas DataFrame to process
    
    **model**: spaCy model to load
    
    **num_topics**: Number of topics to model

    **max_features**: Maximum number of features to consider

    **max_iter**: Maximum number of epochs to fit data

    **min_df**: Minimum length of words

    **max_df**: Maximum length of words

    **worker**: Number of workers

    **colour**: Colour of the plots

    **alpha**: Alpha value

    **l1_ratio**: L1 ratio value
    """

    global fc
    
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
                cv = CountVectorizer(min_df=min_df,
                                     max_df=max_df,
                                     stop_words='english',
                                     lowercase=True,
                                     token_pattern=r'[a-zA-Z\-][a-zA-Z\-]{2,}',
                                     max_features=max_features)
                vectorised = cv.fit_transform(raw_data[data_column])
            except ValueError:
                raise HTTPException(status_code=415, detail='Error: The column loaded is empty or has invalid data'
                                                            ' points. Try again.')
            except Exception as ex:
                raise HTTPException(status_code=415, detail=ex)
            else:
                if model == 'lda':
                    LDA = LatentDirichletAllocation(n_components=num_topics,
                                                    max_iter=max_iter,
                                                    learning_method='online',
                                                    n_jobs=worker)
                    LDA_data = LDA.fit_transform(vectorised)
                    topic_text = modelIterator(LDA, cv, top_n=num_topics,
                                               vb=False)
                    keywords = pd.DataFrame(dominantTopic(vect=cv, model=LDA,
                                                          n_words=num_topics))
                    keywords.columns = [f'word_{i}' for i in range(keywords.shape[1])]
                    keywords.index = [f'topic_{i}' for i in range(keywords.shape[0])]
                    LDA_vis = pyLDAvis.sklearn.prepare(LDA, vectorised, cv, mds='tsne')
                    pyLDAvis.save_html(LDA_vis,
                                       str(os.path.join(os.getcwd(), f'lda_id{fc}.html')))
                    with open(os.path.join(os.getcwd(), f'lda_id{fc}.html')) as f:
                        render = f.read()
                    fc += 1

                    data = {
                        'topic_text': {i: (topic_text[i].to_json()) for i
                                       in range(len(topic_text))},
                        'data': raw_data.to_json(),
                        'keywords': keywords.to_json(),
                        'render': render
                    }

                    return jsonable_encoder(data)

                elif model == 'nmf':
                    TFIDF = TfidfVectorizer(max_df=max_df,
                                            min_df=min_df,
                                            max_features=max_features,
                                            stop_words='english')
                    TFIDF_vectorised = TFIDF.fit_transform(raw_data
                                                           [data_column]
                                                           .values.astype(str))
                    NMF_model = NMF(n_components=num_topics,
                                    max_iter=max_iter,
                                    random_state=1,
                                    alpha=alpha,
                                    l1_ratio=l1_ratio).fit(TFIDF_vectorised)
                    topic_text = modelIterator(model=NMF_model,
                                               vectoriser=TFIDF,
                                               top_n=num_topics,
                                               vb=False)
                    keywords = pd.DataFrame(dominantTopic(model=NMF_model,
                                                          vect=TFIDF,
                                                          n_words=num_topics))
                    keywords.columns = [f'word_{i}' for i in range(keywords.shape[1])]
                    keywords.index = [f'topic_{i}' for i in range(keywords.shape[0])]
                    data = {
                        'topic_text': {i: (topic_text[i].to_json()) for i
                                       in range(len(topic_text))},
                        'data': raw_data.to_json(),
                        'keywords': keywords.to_json()
                    }
                    return jsonable_encoder(data)

                elif model == 'lsi':
                    LSI = TruncatedSVD(n_components=num_topics, n_iter=max_iter)
                    LSI_data = LSI.fit_transform(vectorised)
                    topic_text = modelIterator(LSI, cv,
                                               top_n=num_topics, vb=False)
                    keywords = pd.DataFrame(dominantTopic(model=LSI, vect=cv,
                                                          n_words=num_topics))
                    keywords.columns = [f'word_{i}' for i in range(keywords.shape[1])]
                    keywords.index = [f'topic_{i}' for i in range(keywords.shape[0])]

                    # SVD
                    svd_2d = TruncatedSVD(n_components=2)
                    data_2d = svd_2d.fit_transform(vectorised)

                    mar_fig = go.Scattergl(
                        x=data_2d[:, 0],
                        y=data_2d[:, 1],
                        mode='markers',
                        marker=dict(
                            color=colour,
                            line=dict(width=1)
                        ),
                        text=cv.get_feature_names(),
                        hovertext=cv.get_feature_names(),
                        hoverinfo='text'
                    )
                    mar_fig = [mar_fig]
                    mar_fig = go.Figure(data=mar_fig, layout=go.Layout(title='Scatter Plot'))
                    word_fig = go.Scattergl(
                        x=data_2d[:, 0],
                        y=data_2d[:, 1],
                        mode='text',
                        marker=dict(
                            color=colour,
                            line=dict(width=1)
                        ),
                        text=cv.get_feature_names(),
                    )
                    word_fig = [word_fig]
                    word_fig = go.Figure(data=word_fig, layout=go.Layout(title='Scatter Word Plot'))

                    data = {
                        'topic_text': {i: (topic_text[i].to_json()) for i
                                       in range(len(topic_text))},
                        'data': raw_data.to_json(),
                        'keywords': keywords.to_json(),
                        'point_figure': mar_fig.to_image(format='png'),
                        'word_figure': word_fig.to_image(format='png')
                    }

                    return jsonable_encoder(data)
        else:
            raise HTTPException(status_code=415, detail='Error: Data not loaded properly. Try again.')


@router.post('/classification')
def classification(file: UploadFile = File(...), ftype: str = 'csv', data_column: str = 'data', topics: str = ''):
    """
    Conduct Text Classification


    **file**: Data

    **ftype**: The file format to read the input data as

    **data_column**: Column in the pandas DataFrame to process

    **topics**: A string (delimited by commas) or a list of topics to classify data into
    """

    if torch.cuda.is_available():
        try:
            torch.cuda.get_device_name(torch.cuda.current_device())
        except AssertionError:
            raise HTTPException(status_code=415, detail='Error: CUDA Device is not enabled. Try again.')
        except Exception as ex:
            raise HTTPException(status_code=415, detail=ex)
        else:
            try:
                if ftype == 'csv':
                    raw_data = pd.read_csv(StringIO(str(file.file.read(), 'latin1')), encoding='latin1').astype(object)
                elif ftype == 'xlsx':
                    raw_data = pd.read_excel(StringIO(str(file.file.read(), 'utf-8')), engine='openpyxl').astype(object)
                elif ftype == 'json':
                    raw_data = pd.read_json(StringIO(str(file.file.read(), 'utf-8'))).astype(object)
                else:
                    raise HTTPException(status_code=415, detail='Error: File format input is not supported. Try again.')
            except Exception as ex:
                raise HTTPException(status_code=415, detail=ex)
            else:
                if type(topics) == str:
                    topics = [word.strip().lower() for word in topics.split(sep=',')]
                elif type(topics) == list:
                    topics = topics
                else:
                    raise HTTPException(status_code=415, detail='Error: Invalid data type for topics.')

                classifier = pipeline('zero-shot-classification')
                raw_data['TEST'] = raw_data[data_column].apply(lambda x: classifier(x, topics))
                raw_data['CLASSIFIED'] = raw_data['TEST']. \
                    apply(lambda x: list(zip(x['labels'].tolist(), x['scores'].tolist())))
                raw_data['MOST PROBABLE TOPIC'] = raw_data['CLASSIFIED'].apply(lambda x: max(x, key=itemgetter[1])[0])
                raw_data = raw_data.astype(str)

                data = {
                    'data': raw_data.to_json()
                }
                return jsonable_encoder(data)
    else:
        raise HTTPException(status_code=404, detail='Error: CUDA Device is not detected. Try again.')
