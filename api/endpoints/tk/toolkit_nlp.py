import logging
import os
from collections import Counter
from heapq import nlargest
from string import punctuation
import io
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

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from config import toolkit
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


router = APIRouter(prefix='/endpoints/toolkit',
                   tags=['toolkit'],
                   responses={200: {'description': 'OK'},
                              404: {'description': 'Resource Not Found'},
                              415: {'description': 'Unsupported Media Type'}})


@router.post('/wordcloud')
def wordcloud(json_file, data_column: str = 'data', max_word: int = 200, contour: int = 3, width: int = 800,
              height: int = 400, colour: str = 'steelblue'):
    """
    Wordcloud creation
    
    
    **json_file**: JSON Data

    **data_column**: Column in the pandas DataFrame to process
    
    **max_word**: Max number of words to render in the wordcloud image
    
    **contour**: Contour width
    
    **width**: Width of the wordcloud image
    
    **height**: Height of the wordcloud image
    
    **colour**: Colour of the background
    """
    
    try:
        toolkit['DATA'] = pd.read_json(json_file, low_memory=False, encoding='latin1')
    except Exception as ex:
        raise HTTPException(status_code=415, detail=ex)
    else:
        if not toolkit['DATA'].empty:
            toolkit['DATA'] = toolkit['DATA'][[data_column]]
            wc = WordCloud(background_color='white',
                           max_words=max_word,
                           contour_width=contour,
                           width=width,
                           height=height,
                           contour_color=colour)
            wc.generate(' '.join(toolkit['DATA'][data_column]))
            img = wc.to_image()
            data = {
                'image': str(img.tobytes())
            }

            return jsonable_encoder(data)
        else:
            raise HTTPException(status_code=404, detail='Error: Document-Term Matrix was not properly prepared. Try '
                                                        'again.')


@router.post('/ner')
def ner(json_file, data_column: str = 'data', model='en_core_web_sm', one_datapoint_analyser: int = None):
    """
    Conduct NER analysis
    
    
    **json_file**: JSON Data

    **data_column**: Column in the pandas DataFrame to process
    
    **model**: spaCy model to load
    
    **one_datapoint_analyser**: The datapoint to render into HTML format
    """
    
    try:
        toolkit['DATA'] = pd.read_json(json_file, low_memory=False, encoding='latin1')
    except Exception as ex:
        raise HTTPException(status_code=415, detail=ex)
    else:
        if not toolkit['DATA'].empty:
            # init the required columns
            toolkit['DATA'] = toolkit['DATA'][[data_column]]
            toolkit['DATA']['NER'] = ''
            toolkit['DATA']['COMPILED_LABELS'] = ''
            toolkit['DATA'] = toolkit['DATA'].astype(str)

            if model == 'en_core_web_sm':
                try:
                    toolkit['NLP'] = spacy.load('en_core_web_sm')
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
                    toolkit['NLP'] = spacy.load('en_core_web_lg')
                except OSError:
                    logging.warning('Model not found, downloading...')
                    try:
                        os.system('python -m spacy download en_core_web_lg')
                    except Exception as ex:
                        logging.error(f'Unable to download Model. Error: {ex}')
                        raise HTTPException(status_code=415, detail=ex)
                except Exception as ex:
                    raise HTTPException(status_code=415, detail=ex)

            for index in range(len(toolkit['DATA'])):
                temp_nlp = toolkit['NLP'](toolkit['DATA'][toolkit['DATA_COLUMN']][index])
                toolkit['DATA'].at[index, 'NER'] = str(list(zip([word.text for word in temp_nlp.ents],
                                                                [word.label_ for word in temp_nlp.ents])))
                toolkit['DATA'].at[index, 'COMPILED_LABELS'] = str(list(set([word.label_ for word
                                                                             in temp_nlp.ents])))

            if one_datapoint_analyser is not None:
                cpy = toolkit['DATA'].copy()
                temp = cpy[data_column][one_datapoint_analyser]
                render = displacy.render(list(toolkit['NLP'](str(temp)).sents),
                                         style='ent',
                                         page=True)
                data = {
                    'data': toolkit['DATA'].to_json(),
                    'render': render
                }
                return jsonable_encoder(data)
            else:
                data = {
                    'data': toolkit['DATA'].to_json()
                }
                return jsonable_encoder(data)
        else:
            raise HTTPException(status_code=415, detail='Error: Data not loaded properly. Try again.')


@router.post('/pos')
def pos(json_file, data_column: str = '', model='en_core_web_sm', one_datapoint_analyser: int = None,
        compact: bool = True, colour: str = 'steelblue', bg: str = 'white'):
    """
    Conduct POS tagging
    
    
    **json_file**: JSON Data

    **data_column**: Column in the pandas DataFrame to process
    
    **model**: spaCy model to load
    
    **one_datapoint_analyser**: The datapoint to render into HTML format
    
    **compact**: Compact the renders
    
    **colour**: Colour of the words in the render
    
    **bg**: Colour of the background
    """
    
    try:
        toolkit['DATA'] = pd.read_json(json_file, low_memory=False, encoding='latin1')
    except Exception as ex:
        raise HTTPException(status_code=415, detail=ex)
    else:
        if not toolkit['DATA'].empty:
            # init the required columns
            toolkit['DATA'] = toolkit['DATA'][[data_column]]
            toolkit['DATA']['POS'] = ''
            toolkit['DATA'] = toolkit['DATA'].astype(str)

            if model == 'en_core_web_sm':
                try:
                    toolkit['NLP'] = spacy.load('en_core_web_sm')
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
                    toolkit['NLP'] = spacy.load('en_core_web_lg')
                except OSError:
                    logging.warning('Model not found, downloading...')
                    try:
                        os.system('python -m spacy download en_core_web_lg')
                    except Exception as ex:
                        logging.error(f'Unable to download Model. Error: {ex}')
                        raise HTTPException(status_code=415, detail=ex)
                except Exception as ex:
                    raise HTTPException(status_code=415, detail=ex)

            for index in range(len(toolkit['DATA'])):
                temp_nlp = toolkit['NLP'](toolkit['DATA'][toolkit['DATA_COLUMN']][index])
                toolkit['DATA'].at[index, 'POS'] = str(list(zip([str(word) for word in temp_nlp],
                                                                [word.pos_ for word in temp_nlp])))
                toolkit['DATA'].at[index, 'COMPILED_LABELS'] = str(list(set([word.pos_ for word in temp_nlp])))

            if one_datapoint_analyser is not None:
                cpy = toolkit['DATA'].copy()
                temp = cpy[data_column][one_datapoint_analyser]
                render = displacy.render(list(toolkit['NLP'](str(temp)).sents),
                                         style='dep',
                                         options={
                                             'compact': compact,
                                             'color': colour,
                                             'bg': bg,
                                         })
                data = {
                    'data': toolkit['DATA'].to_json(),
                    'render': render
                }
                return jsonable_encoder(data)
            else:
                data = {
                    'data': toolkit['DATA'].to_json()
                }
                return jsonable_encoder(data)
        else:
            raise HTTPException(status_code=415, detail='Error: Data not loaded properly. Try again.')


@router.post('/summarise')
def summarise(json_file, data_column: str = 'data', mode: str = 'basic', model: str = 'en_core_web_sm',
              sentence_len: int = 3, min_words: int = 80, max_words: str = 150, max_tensor: int = 512):
    """
    Summarise texts
    
    
    **json_file**: JSON Data

    **data_column**: Column in the pandas DataFrame to process
    
    **mode**: Define whether or not to conduct 'basic' or 'advanced' summarisation on input data
    
    **model**: spaCy model to load
    
    **sentence_len**: The maximum length of sentence to return
    
    **min_words**: The minimum number of words to include in the summary
    
    **max_words**: The maximum number of words to include in the summary
    
    **max_tensor**: The maximum number of input tensors for advanced summarisation process
    """
    
    try:
        toolkit['DATA'] = pd.read_json(json_file, low_memory=False, encoding='latin1')
    except Exception as ex:
        raise HTTPException(status_code=415, detail=ex)
    else:
        # load up the data first
        toolkit['DATA'] = toolkit['DATA'][[data_column]]
        toolkit['DATA']['SUMMARY'] = np.nan
        toolkit['DATA'] = toolkit['DATA'].astype(str)

        if not toolkit['DATA'].empty:
            if mode == 'basic':
                if model == 'en_core_web_sm':
                    try:
                        toolkit['NLP'] = spacy.load('en_core_web_sm')
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
                        toolkit['NLP'] = spacy.load('en_core_web_lg')
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
                toolkit['DATA']['SUMMARY'] = toolkit['DATA'][toolkit['DATA_COLUMN']]. \
                    apply(lambda x: summarise(x, stopwords, pos_tag, toolkit['NLP'], sentence_len))
                data = {
                    'data': toolkit['DATA'].to_json()
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
                        toolkit['DATA'] = toolkit['DATA'].astype(object)
                        toolkit['DATA']['ENCODED'] = toolkit['DATA'][data_column]. \
                            apply(lambda x: tokenizer.encode('summarize: ' + x,
                                                             return_tensors='pt',
                                                             max_length=max_tensor,
                                                             truncation=True))
                        toolkit['DATA']['OUTPUTS'] = toolkit['DATA']['ENCODED']. \
                            apply(lambda x: model.generate(x,
                                                           max_length=max_words,
                                                           min_length=min_words,
                                                           length_penalty=5.0,
                                                           num_beams=2))
                        toolkit['DATA']['SUMMARISED'] = toolkit['DATA']['OUTPUTS'].apply(
                            lambda x: tokenizer.decode(x[0]))
                        toolkit['DATA'].drop(columns=['ENCODED', 'OUTPUTS'], inplace=True)
                        toolkit['DATA']['SUMMARISED'] = toolkit['DATA']['SUMMARISED']. \
                            str.replace('<pad> ', '').str.replace('</s>', '')
                        toolkit['DATA'] = toolkit['DATA'].astype(str)
                        data = {
                            'data': toolkit['DATA'].to_json()
                        }
                        return jsonable_encoder(data)

        else:
            raise HTTPException(status_code=415, detail='Error: Data not loaded properly. Try again.')


@router.post('/sentiment')
def sentiment(json_file, data_column: str = 'data', model='vader', colour='#2ACAEA'):
    """
    Conduct Sentiment Analysis
    
    
    **json_file**: JSON Data

    **data_column**: Column in the pandas DataFrame to process
    
    **model**: spaCy model to load
    
    **colour**: Colour of plots generated
    """
    
    try:
        toolkit['DATA'] = pd.read_json(json_file, low_memory=False, encoding='latin1')
    except Exception as ex:
        raise HTTPException(status_code=415, detail=ex)
    else:
        if not toolkit['DATA'].empty:
            if model == 'vader':
                replacer = {
                    r"'": '',
                    r'[^\w\s]': ' ',
                    r' \d+': ' ',
                    r' +': ' '
                }

                toolkit['DATA']['VADER SENTIMENT TEXT'] = toolkit['DATA'][data_column]. \
                    replace(to_replace=replacer, regex=True)

                vader_analyser = SentimentIntensityAnalyzer()
                sent_score_list = []
                sent_label_list = []

                # scoring
                for i in toolkit['DATA']['VADER SENTIMENT TEXT'].tolist():
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

                toolkit['DATA']['VADER OVERALL SENTIMENT'] = sent_label_list
                toolkit['DATA']['VADER OVERALL SCORE'] = sent_score_list
                toolkit['DATA']['VADER POSITIVE SCORING'] = [vader_analyser.polarity_scores(doc)['pos'] for doc in
                                                             toolkit['DATA']['VADER SENTIMENT TEXT']
                                                                 .values.tolist()]
                toolkit['DATA']['VADER NEUTRAL SCORING'] = [vader_analyser.polarity_scores(doc)['neu'] for doc in
                                                            toolkit['DATA']['VADER SENTIMENT TEXT'].values.tolist()]
                toolkit['DATA']['VADER NEGATIVE SCORING'] = [vader_analyser.polarity_scores(doc)['neg'] for doc in
                                                             toolkit['DATA']['VADER SENTIMENT TEXT']
                                                                 .values.tolist()]

                # create plots
                toolkit['HAC_PLOT'] = ff.create_distplot([toolkit['DATA']['VADER OVERALL SCORE'].tolist()],
                                                         ['VADER'],
                                                         colors=[colour],
                                                         bin_size=0.25,
                                                         curve_type='normal',
                                                         show_rug=False,
                                                         show_hist=False)
                toolkit['HAC_PLOT'].update_layout(title_text='Distribution Plot',
                                                  xaxis_title='VADER Score',
                                                  yaxis_title='Frequency Density',
                                                  legend_title='Frequency Density')
                data = {
                    'data': toolkit['DATA'].to_json(),
                    'image': toolkit['HAC_PLOT'].to_image(format="png")

                }
                return jsonable_encoder(data)

            elif model == 'textblob':
                pol_list = []
                sub_list = []

                # scoring: polarity
                toolkit['DATA']['POLARITY SCORE'] = toolkit['DATA'][data_column]. \
                    apply(lambda x: TextBlob(x).sentiment.polarity)
                for i in toolkit['DATA']['POLARITY SCORE'].tolist():
                    if float(i) > 0:
                        pol_list.append('Positive')
                    elif float(i) < 0:
                        pol_list.append('Negative')
                    elif float(i) == 0:
                        pol_list.append('Neutral')
                toolkit['DATA']['POLARITY SENTIMENT'] = pol_list

                # scoring: subjectivity
                toolkit['DATA']['SUBJECTIVITY SCORE'] = toolkit['DATA'][data_column].apply(
                    lambda x: TextBlob(x).sentiment.subjectivity
                )
                for i in toolkit['DATA']['SUBJECTIVITY SCORE'].tolist():
                    if float(i) < 0.5:
                        sub_list.append('Objective')
                    elif float(i) > 0.5:
                        sub_list.append('Subjective')
                    elif float(i) == 0.5:
                        sub_list.append('Neutral')
                toolkit['DATA']['SUBJECTIVITY SENTIMENT'] = sub_list
                toolkit['HAC_PLOT'] = px.scatter(toolkit['DATA'][['SUBJECTIVITY SCORE', 'POLARITY SCORE']],
                                                 x='SUBJECTIVITY SCORE',
                                                 y='POLARITY SCORE',
                                                 labels={
                                                     'SUBJECTIVITY SCORE': 'Subjectivity',
                                                     'POLARITY SCORE': 'Polarity'
                                                 })
                toolkit['HAC_PLOT1'] = ff.create_distplot([toolkit['DATA']['SUBJECTIVITY SCORE'].tolist(),
                                                           toolkit['DATA']['POLARITY SCORE'].tolist()],
                                                          ['Subjectivity', 'Polarity'],
                                                          curve_type='normal',
                                                          show_rug=False,
                                                          show_hist=False)
                data = {
                    'data': toolkit['DATA'].to_json(),
                    'image': toolkit['HAC_PLOT'].to_image(format="png"),
                    'image1': toolkit['HAC_PLOT1'].to_image(format="png")
                }
                return jsonable_encoder(data)
        else:
            raise HTTPException(status_code=415, detail='Error: Data not loaded properly. Try again.')


@router.post
def topic_modelling(json_file, data_column: str = 'data', model: str = 'lda', num_topics: int = 10,
                    max_features: int = 5000, max_iter: int = 10, min_df: int = 5,
                    max_df: float = 0.90, worker: int = 1, alpha: float = 0.10,
                    l1_ratio: float = 0.50):
    """
    Topic Modelling
    
    
    **json_file**: JSON Data

    **data_column**: Column in the pandas DataFrame to process
    
    **model**: spaCy model to load
    
    **num_topics**: Number of topics to model

    **max_features**: Maximum number of features to consider

    **max_iter**: Maximum number of epochs to fit data

    **min_df**: Minimum length of words

    **max_df**: Maximum length of words

    **worker**: Number of workers

    **alpha**: Alpha value

    **l1_ratio**: L1 ratio value
    """

    try:
        toolkit['DATA'] = pd.read_json(json_file, low_memory=False, encoding='latin1')
    except Exception as ex:
        raise HTTPException(status_code=415, detail=ex)
    else:
        if not toolkit['DATA'].empty:
            if model == 'lda':
                try:
                    toolkit['CV'] = CountVectorizer(min_df=min_df,
                                                    max_df=max_df,
                                                    stop_words='english',
                                                    lowercase=True,
                                                    token_pattern=r'[a-zA-Z\-][a-zA-Z\-]{2,}',
                                                    max_features=max_features)
                    toolkit['VECTORISED'] = toolkit['CV'].fit_transform(toolkit['DATA'][data_column])
                except ValueError:
                    raise HTTPException(status_code=415, detail='Error: The column loaded is empty or has invalid data'
                                                                ' points. Try again.')
                except Exception as ex:
                    raise HTTPException(status_code=415, detail=ex)
                else:
                    toolkit['LDA_MODEL'] = LatentDirichletAllocation(n_components=num_topics,
                                                                     max_iter=max_iter,
                                                                     learning_method='online',
                                                                     n_jobs=worker)
                    toolkit['LDA_DATA'] = toolkit['LDA_MODEL'].fit_transform(toolkit['VECTORISED'])
                    toolkit['TOPIC_TEXT'] = modelIterator(toolkit['LDA_MODEL'], toolkit['CV'], top_n=num_topics,
                                                          vb=False)
                    toolkit['KW'] = pd.DataFrame(dominantTopic(vect=toolkit['CV'], model=toolkit['LDA_MODEL'],
                                                               n_words=num_topics))
                    toolkit['KW'].columns = [f'word_{i}' for i in range(toolkit["KW"].shape[1])]
                    toolkit['KW'].index = [f'topic_{i}' for i in range(toolkit["KW"].shape[0])]
                    toolkit['LDA_VIS'] = pyLDAvis.sklearn.prepare(toolkit['LDA_MODEL'], toolkit['VECTORISED'],
                                                                  toolkit['CV'], mds='tsne')
                    pyLDAvis.save_html(toolkit['LDA_VIS'],
                                       str(os.path.join(os.getcwd(), f'lda_id{toolkit["FC"]}.html')))
                    toolkit['FC'] += 1
                    with open(os.path.join(os.getcwd(), f'lda_id{toolkit["FC"]}.html')) as f:
                        render = f.read()

                    data = {
                        'topic_text': {i: (toolkit['TOPIC_TEXT'][i].to_json()) for i
                                       in range(len(toolkit['TOPIC_TEXT']))},
                        'data': toolkit['DATA'].to_json(),
                        'keywords': toolkit['KW'].to_json(),
                        'render': render
                    }

                    return jsonable_encoder(data)

            elif model == 'nmf':
                toolkit['TFIDF_MODEL'] = TfidfVectorizer(max_df=max_df,
                                                         min_df=min_df,
                                                         max_features=max_features,
                                                         stop_words='english')
                toolkit['TFIDF_VECTORISED'] = toolkit['TFIDF_MODEL'].fit_transform(toolkit['DATA']
                                                                                   [data_column]
                                                                                   .values.astype(str))
                toolkit['NMF_MODEL'] = NMF(n_components=num_topics,
                                           max_iter=max_iter,
                                           random_state=1,
                                           alpha=alpha,
                                           l1_ratio=l1_ratio).fit(toolkit['TFIDF_VECTORISED'])
                toolkit['TOPIC_TEXT'] = modelIterator(model=toolkit['NMF_MODEL'],
                                                      vectoriser=toolkit['TFIDF_MODEL'],
                                                      top_n=toolkit['NUM_TOPICS'],
                                                      vb=False)
                toolkit['KW'] = pd.DataFrame(dominantTopic(model=toolkit['NMF_MODEL'],
                                                           vect=toolkit['TFIDF_MODEL'],
                                                           n_words=num_topics))
                toolkit['KW'].columns = [f'word_{i}' for i in range(toolkit['KW'].shape[1])]
                toolkit['KW'].index = [f'topic_{i}' for i in range(toolkit['KW'].shape[0])]
                data = {
                    'topic_text': {i: (toolkit['TOPIC_TEXT'][i].to_json()) for i
                                   in range(len(toolkit['TOPIC_TEXT']))},
                    'data': toolkit['DATA'].to_json(),
                    'keywords': toolkit['KW'].to_json()
                }
                return jsonable_encoder(data)

            elif model == 'lsi':
                toolkit['LSI_MODEL'] = TruncatedSVD(n_components=num_topics,
                                                    n_iter=max_iter)
                toolkit['LSI_DATA'] = toolkit['LSI_MODEL'].fit_transform(toolkit['VECTORISED'])
                toolkit['TOPIC_TEXT'] = modelIterator(toolkit['LSI_MODEL'], toolkit['CV'],
                                                      top_n=num_topics, vb=False)
                toolkit['KW'] = pd.DataFrame(dominantTopic(model=toolkit['LSI_MODEL'], vect=toolkit['CV'],
                                                           n_words=num_topics))
                toolkit['KW'].columns = [f'word_{i}' for i in range(toolkit['KW'].shape[1])]
                toolkit['KW'].index = [f'topic_{i}' for i in range(toolkit['KW'].shape[0])]

                # SVD
                svd_2d = TruncatedSVD(n_components=2)
                data_2d = svd_2d.fit_transform(toolkit['VECTORISED'])

                toolkit['MAR_FIG'] = go.Scattergl(
                    x=data_2d[:, 0],
                    y=data_2d[:, 1],
                    mode='markers',
                    marker=dict(
                        color=toolkit['COLOUR'],
                        line=dict(width=1)
                    ),
                    text=toolkit['CV'].get_feature_names(),
                    hovertext=toolkit['CV'].get_feature_names(),
                    hoverinfo='text'
                )
                toolkit['MAR_FIG'] = [toolkit['MAR_FIG']]
                toolkit['MAR_FIG'] = go.Figure(data=toolkit['MAR_FIG'],
                                               layout=go.Layout(title='Scatter Plot'))
                toolkit['WORD_FIG'] = go.Scattergl(
                    x=data_2d[:, 0],
                    y=data_2d[:, 1],
                    mode='text',
                    marker=dict(
                        color=toolkit['COLOUR'],
                        line=dict(width=1)
                    ),
                    text=toolkit['CV'].get_feature_names(),
                )
                toolkit['WORD_FIG'] = [toolkit['WORD_FIG']]
                toolkit['WORD_FIG'] = go.Figure(data=toolkit['WORD_FIG'],
                                                layout=go.Layout(title='Scatter Word Plot'))

                data = {
                    'topic_text': {i: (toolkit['TOPIC_TEXT'][i].to_json()) for i
                                   in range(len(toolkit['TOPIC_TEXT']))},
                    'data': toolkit['DATA'].to_json(),
                    'keywords': toolkit['KW'].to_json(),
                    'point_figure': toolkit['MAR_FIG'].to_image(format='png'),
                    'word_figure': toolkit['WORD_FIG'].to_image(format='png')
                }

                return jsonable_encoder(data)
        else:
            raise HTTPException(status_code=415, detail='Error: Data not loaded properly. Try again.')


@router.post('/classification')
def classification(json_file, data_column: str = 'data', topics=''):
    """
    Conduct Text Classification


    **json_file**: JSON Data

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
                toolkit['DATA'] = pd.read_json(json_file, low_memory=False, encoding='latin1')
            except Exception as ex:
                raise HTTPException(status_code=415, detail=ex)
            else:
                if type(topics) == str:
                    toolkit['CLASSIFY_TOPIC'] = [word.strip().lower() for word in toolkit['CLASSIFY_TOPIC']
                        .split(sep=',')]
                elif type(topics) == list:
                    toolkit['CLASSIFY_TOPIC'] = topics
                else:
                    raise HTTPException(status_code=415, detail='Error: Invalid data type for topics.')

                toolkit['DATA'] = toolkit['DATA'].astype(object)
                classifier = pipeline('zero-shot-classification')
                toolkit['DATA']['TEST'] = toolkit['DATA'][data_column]. \
                    apply(lambda x: classifier(x, toolkit['CLASSIFY_TOPIC']))
                toolkit['DATA']['CLASSIFIED'] = toolkit['DATA']['TEST']. \
                    apply(lambda x: list(zip(x['labels'].tolist(), x['scores'].tolist())))
                toolkit['DATA']['MOST PROBABLE TOPIC'] = toolkit['DATA']['CLASSIFIED']. \
                    apply(lambda x: max(x, key=itemgetter[1])[0])
                toolkit['DATA'] = toolkit['DATA'].astype(str)

                data = {
                    'data': toolkit['DATA'].to_json()
                }
                return jsonable_encoder(data)
    else:
        raise HTTPException(status_code=404, detail='Error: CUDA Device is not detected. Try again.')
