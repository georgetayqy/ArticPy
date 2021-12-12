"""
This module allows the user to train models and to predict NLP data
"""

# -------------------------------------------------------------------------------------------------------------------- #
# |                                         IMPORT RELEVANT LIBRARIES                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
import multiprocessing
import os
import pathlib
import matplotlib
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
import streamlit.components.v1
import textattack.models.wrappers
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import transformers

from config import toolkit, STREAMLIT_STATIC_PATH, DOWNLOAD_PATH
from operator import itemgetter
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, AutoModelForSequenceClassification
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from streamlit_pandas_profiling import st_profile_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from spacy import displacy
from wordcloud import WordCloud
from textblob import TextBlob
from utils import csp_downloaders
from utils.helper import readFile, summarise, modelIterator, printDataFrame, dominantTopic


# -------------------------------------------------------------------------------------------------------------------- #
# |                                             MAIN APP FUNCTIONALITY                                               | #
# -------------------------------------------------------------------------------------------------------------------- #
def app():
    """
    Main function that will be called when the app is run
    """

    # -------------------------------------------------------------------------------------------------------------------- #
    # |                                                    INIT                                                          | #
    # -------------------------------------------------------------------------------------------------------------------- #
    # CREATE THE DOWNLOAD PATH WHERE FILES CREATED WILL BE STORED AND AVAILABLE FOR DOWNLOADING
    if not DOWNLOAD_PATH.is_dir():
        DOWNLOAD_PATH.mkdir()

    st.markdown('# NLP Model Trainer and Predictor')
    st.markdown('This function allows you to train and create a ML Model to classify the topic of the News Article '
                'passed on to the dataset. This function requires the use of the PyTorch Library to train and '
                'evaluate your model. Ensure that you have downloaded and installed the correct PyTorch library '
                'corresponding to your CUDA version.')

    st.markdown('---')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('### PyTorch for CUDA 10.2')
        if st.button('Install Relevant Packages', key='10.2'):
            os.system('pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102'
                      ' -f https://download.pytorch.org/whl/cu102/torch_stable.html')
    with col2:
        st.markdown('### PyTorch for CUDA 11.3')
        if st.button('Install Relevant Packages', key='11.3'):
            os.system('pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113'
                      ' -f https://download.pytorch.org/whl/cu113/torch_stable.html')
    st.markdown('\n\n')

    if st.button('Check if GPU is properly installed'):
        st.info(f'GPU Installation Status: **{torch.cuda.is_available()}**')
    if st.button('Check GPU used'):
        try:
            st.info(f'GPU Device **{torch.cuda.get_device_name(torch.cuda.current_device())}** in use.')
        except AssertionError:
            st.error('Your version of PyTorch is CPU-optimised. Download and install any of the above two '
                     'supported GPU-enabled PyTorch versions to use your GPU and silence this error.')
        except Exception as ex:
            st.error(ex)

    st.markdown('---')
    st.markdown('## Mode Selector')
    toolkit['MODEL_MODE'] = st.selectbox('Select the actions you want to perform', ('Training', 'Evaluation'))

    if toolkit['MODEL_MODE'] == 'Training':
        st.markdown('## Flags\n\n'
                    '### Training Parameters')
        toolkit['API'] = st.checkbox('Use Training API?',
                                     help='Note that with this option selected, you must ensure that your GPU has '
                                          'sufficient GPU memory to run the networks/models you selected. If you '
                                          'are unsure, it is better to use the Command Line Argument API to fine '
                                          'tune the model parameters before starting the training.',
                                     value=True)

        if toolkit['API']:
            toolkit['TRAINING_PARAMS'] = st.multiselect('Select Training Parameters',
                                                        ('num_epochs', 'num_clean_epochs', 'attack_epoch_interval',
                                                         'early_stopping_epochs', 'learning_rate',
                                                         'num_warmup_steps',
                                                         'weight_decay', 'per_device_train_batch_size',
                                                         'per_device_eval_batch_size',
                                                         'gradient_accumulation_steps', 'random_seed', 'parallel',
                                                         'load_best_model_at_end', 'alpha',
                                                         'num_train_adv_examples', 'query_budget_train',
                                                         'attack_num_workers_per_device', 'output_dir',
                                                         'checkpoint_interval_steps', 'checkpoint_interval_epochs',
                                                         'save_last', 'log_to_tb', 'tb_log_dir', 'log_to_wandb',
                                                         'wandb_project', 'logging_interval_step'),
                                                        default=('num_epochs', 'per_device_train_batch_size'))
            if 'num_epochs' in toolkit['TRAINING_PARAMS']:
                toolkit['num_epochs'] = st.number_input('Total number of epochs for training',
                                                        min_value=1,
                                                        max_value=1000000,
                                                        value=3,
                                                        key='num_epochs')
            if 'num_clean_epochs' in toolkit['TRAINING_PARAMS']:
                toolkit['num_clean_epochs'] = st.number_input('Number of epochs to train on just the original '
                                                              'training dataset before adversarial training',
                                                              min_value=1,
                                                              max_value=1000000,
                                                              value=1,
                                                              key='num_clean_epochs')
            if 'attack_epoch_interval' in toolkit['TRAINING_PARAMS']:
                toolkit['attack_epoch_interval'] = st.number_input('Generate a new adversarial training set every '
                                                                   'N epochs',
                                                                   min_value=1,
                                                                   max_value=1000000,
                                                                   value=1,
                                                                   key='attack_epoch_interval')
            if 'early_stopping_epochs' in toolkit['TRAINING_PARAMS']:
                toolkit['early_stopping_epochs'] = st.number_input('Number of epochs validation must increase '
                                                                   'before stopping early',
                                                                   min_value=1,
                                                                   max_value=1000000,
                                                                   value=1,
                                                                   key='early_stopping_epochs')
            else:
                toolkit['early_stopping_epochs'] = None
            if 'learning_rate' in toolkit['TRAINING_PARAMS']:
                toolkit['learning_rate'] = st.number_input('Number of epochs validation must increase before '
                                                           'stopping early',
                                                           min_value=0,
                                                           max_value=1,
                                                           value=5e-5,
                                                           step=0.000001,
                                                           format='.%6f',
                                                           key='learning_rate')
            if 'num_warmup_steps' in toolkit['TRAINING_PARAMS']:
                if st.checkbox('Define in float?'):
                    toolkit['num_warmup_steps'] = st.number_input('The number of steps for the warmup phase of '
                                                                  'linear scheduler',
                                                                  min_value=0,
                                                                  max_value=1,
                                                                  value=0.50,
                                                                  step=0.001,
                                                                  format='.%3f',
                                                                  key='num_warmup_steps')
                else:
                    toolkit['num_warmup_steps'] = st.number_input('The number of steps for the warmup phase of '
                                                                  'linear scheduler',
                                                                  min_value=1,
                                                                  max_value=1000000,
                                                                  value=500,
                                                                  key='num_warmup_steps')
            if 'weight_decay' in toolkit['TRAINING_PARAMS']:
                toolkit['weight_decay'] = st.number_input('Weight decay (L2 penalty)',
                                                          min_value=0,
                                                          max_value=1,
                                                          value=0.01,
                                                          step=0.01,
                                                          format='.%2f',
                                                          key='weight_decay')
            if 'per_device_train_batch_size' in toolkit['TRAINING_PARAMS']:
                toolkit['per_device_train_batch_size'] = st.number_input('The batch size per GPU/CPU for training',
                                                                         min_value=1,
                                                                         max_value=1000000,
                                                                         value=8,
                                                                         key='per_device_train_batch_size')
            if 'per_device_eval_batch_size' in toolkit['TRAINING_PARAMS']:
                toolkit['per_device_eval_batch_size'] = st.number_input('The batch size per GPU/CPU for evaluation',
                                                                        min_value=1,
                                                                        max_value=1000000,
                                                                        value=32,
                                                                        key='per_device_eval_batch_size')
            if 'gradient_accumulation_steps' in toolkit['TRAINING_PARAMS']:
                toolkit['gradient_accumulation_steps'] = st.number_input('Number of updates steps to accumulate '
                                                                         'the gradients before performing a '
                                                                         'backward/update pass',
                                                                         min_value=1,
                                                                         max_value=1000000,
                                                                         value=32,
                                                                         key='gradient_accumulation_steps')
            if 'random_seed' in toolkit['TRAINING_PARAMS']:
                toolkit['random_seed'] = st.number_input('Random seed for reproducibility',
                                                         min_value=1,
                                                         max_value=1000000,
                                                         value=32,
                                                         key='random_seed')
            if 'parallel' in toolkit['TRAINING_PARAMS']:
                toolkit['parallel'] = st.checkbox('Use Multiple GPUs using torch.DataParallel class?',
                                                  value=False,
                                                  key='parallel')
            if 'load_best_model_at_end' in toolkit['TRAINING_PARAMS']:
                toolkit['load_best_model_at_end'] = st.checkbox('keep track of the best model across training and '
                                                                'load it at the end',
                                                                value=False,
                                                                key='parallel')
            if 'alpha' in toolkit['TRAINING_PARAMS']:
                toolkit['alpha'] = st.number_input('The weight for adversarial loss',
                                                   min_value=0,
                                                   max_value=1,
                                                   value=0.50,
                                                   step=0.001,
                                                   format='.%3f',
                                                   key='alpha')
            if 'num_train_adv_examples' in toolkit['TRAINING_PARAMS']:
                if st.checkbox('Use Float Parameters?'):
                    toolkit['num_train_adv_examples'] = st.number_input('The number of samples to successfully '
                                                                        'attack when generating adversarial '
                                                                        'training set before start of every epoch',
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        value=0.50,
                                                                        step=0.001,
                                                                        format='.%3f',
                                                                        key='num_train_adv_examples')
                else:
                    toolkit['num_train_adv_examples'] = st.number_input('The number of samples to successfully '
                                                                        'attack when generating adversarial '
                                                                        'training set before start of every epoch',
                                                                        min_value=1,
                                                                        max_value=1000000,
                                                                        value=8,
                                                                        key='per_device_train_batch_size')
            if 'query_budget_train' in toolkit['TRAINING_PARAMS']:
                if st.checkbox('Set Max Query Budget?', value=False):
                    toolkit['query_budget_train'] = st.number_input('The max query budget to use when generating '
                                                                    'adversarial training set',
                                                                    min_value=1,
                                                                    max_value=1000000,
                                                                    value=1,
                                                                    key='query_budget_train')
                else:
                    toolkit['query_budget_train'] = None
            if 'attack_num_workers_per_device' in toolkit['TRAINING_PARAMS']:
                if st.checkbox('Set Number of Worker Process to run attack?', value=False):
                    toolkit['attack_num_workers_per_device'] = st.number_input('Number of worker processes to run '
                                                                               'per device for attack',
                                                                               min_value=1,
                                                                               max_value=1000000,
                                                                               value=1,
                                                                               key='attack_num_workers_per_device')
                else:
                    toolkit['attack_num_workers_per_device'] = 1
            if 'output_dir' in toolkit['TRAINING_PARAMS']:
                dt = datetime.now()
                toolkit['output_dir'] = st.text_input('Directory to output training logs and checkpoints',
                                                      value=f'/outputs/{dt.strftime("%Y-%m-%d-%H-%M-%S-%f")}',
                                                      key='output_dir')
            if 'checkpoint_interval_steps' in toolkit['TRAINING_PARAMS']:
                if st.checkbox('Save Model Checkpoint after every N updates?'):
                    toolkit['checkpoint_interval_steps'] = st.number_input('Save after N updates',
                                                                           min_value=1,
                                                                           max_value=1000000,
                                                                           value=1,
                                                                           key='checkpoint_interval_steps')
                else:
                    toolkit['checkpoint_interval_steps'] = None
            if 'checkpoint_interval_epochs' in toolkit['TRAINING_PARAMS']:
                if st.checkbox('Save Model Checkpoint after every N epochs?'):
                    toolkit['checkpoint_interval_epochs'] = st.number_input('Save after N epochs',
                                                                            min_value=1,
                                                                            max_value=1000000,
                                                                            value=1,
                                                                            key='checkpoint_interval_epochs')
                else:
                    toolkit['checkpoint_interval_epochs'] = None
            if 'save_last' in toolkit['TRAINING_PARAMS']:
                toolkit['save_last'] = st.checkbox('Save the model at end of training',
                                                   value=True,
                                                   key='save_last')
            if 'log_to_tb' in toolkit['TRAINING_PARAMS']:
                toolkit['log_to_tb'] = st.checkbox('Log to Tensorboard',
                                                   value=False,
                                                   key='log_to_tb')
            if 'tb_log_dir' in toolkit['TRAINING_PARAMS']:
                toolkit['tb_log_dir'] = st.text_input('Directory to output training logs and checkpoints',
                                                      value=r'./runs',
                                                      key='tb_log_dir')
            if 'log_to_wandb' in toolkit['TRAINING_PARAMS']:
                toolkit['log_to_wandb'] = st.checkbox('Log to Wandb',
                                                      value=False,
                                                      key='log_to_wandb')
            if 'wandb_project' in toolkit['TRAINING_PARAMS']:
                toolkit['wandb_project'] = st.text_input('Name of Wandb project for logging',
                                                         value=r'textattack',
                                                         key='wandb_project')
            if 'logging_interval_step' in toolkit['TRAINING_PARAMS']:
                toolkit['logging_interval_step'] = st.number_input('Log to Tensorboard/Wandb every N training '
                                                                   'steps',
                                                                   min_value=1,
                                                                   max_value=1000000,
                                                                   value=1,
                                                                   key='logging_interval_step')
        else:
            toolkit['TRAINING_PARAMS'] = st.multiselect('Select Training Parameters',
                                                        ('attack', 'model_max_length',
                                                         'model_num_labels', 'dataset_train_split',
                                                         'dataset_eval_split', 'filter_train_by_labels',
                                                         'filter_eval_by_labels', 'num_epochs', 'num_clean_epochs',
                                                         'attack_epoch_interval', 'early_stopping_epochs',
                                                         'learning_rate', 'num_warmup_steps',
                                                         'weight_decay', 'per_device_train_batch_size',
                                                         'per_device_eval_batch_size',
                                                         'gradient_accumulation_steps', 'random_seed', 'parallel',
                                                         'load_best_model_at_end', 'alpha',
                                                         'num_train_adv_examples', 'query_budget_train',
                                                         'attack_num_workers_per_device', 'output_dir',
                                                         'checkpoint_interval_steps', 'checkpoint_interval_epochs',
                                                         'save_last', 'log_to_tb', 'tb_log_dir', 'log_to_wandb',
                                                         'wandb_project', 'logging_interval_step'),
                                                        default=('model_max_length', 'num_epochs',
                                                                 'per_device_train_batch_size'))
            if 'attack' in toolkit['TRAINING_PARAMS']:
                toolkit['num_epochs'] = st.text_input('Attack string', key='attack')
            if 'model_max_length' in toolkit['TRAINING_PARAMS']:
                if st.checkbox('Define Model Max Length'):
                    toolkit['model_max_length'] = st.number_input('Model Max Length',
                                                                  min_value=1,
                                                                  max_value=1000000,
                                                                  value=64,
                                                                  key='model_max_length')
                else:
                    toolkit['model_max_length'] = None
            if 'model_num_labels' in toolkit['TRAINING_PARAMS']:
                if st.checkbox('Define Number of Labels'):
                    toolkit['model_num_labels'] = st.number_input('Number of Labels',
                                                                  min_value=1,
                                                                  max_value=1000000,
                                                                  value=1,
                                                                  key='model_num_labels')
                else:
                    toolkit['model_num_labels'] = None
            if 'filter_train_by_labels' in toolkit['TRAINING_PARAMS']:
                toolkit['filter_train_by_labels'] = st.text_input('Filter Train Data By Labels',
                                                                  key='filter_train')
                toolkit['filter_train_by_labels'] = [
                    label for label in toolkit['filter_train_by_labels'].split(',')
                ]
            if 'filter_eval_by_labels' in toolkit['TRAINING_PARAMS']:
                toolkit['filter_eval_by_labels'] = st.text_input('Filter Test Data By Labels',
                                                                 key='filter_test')
                toolkit['filter_eval_by_labels'] = [
                    label for label in toolkit['filter_eval_by_labels'].split(',')
                ]
            if 'num_epochs' in toolkit['TRAINING_PARAMS']:
                toolkit['num_epochs'] = st.number_input('Total number of epochs for training',
                                                        min_value=1,
                                                        max_value=1000000,
                                                        value=3,
                                                        key='num_epochs')
            if 'num_clean_epochs' in toolkit['TRAINING_PARAMS']:
                toolkit['num_clean_epochs'] = st.number_input('Number of epochs to train on just the original '
                                                              'training dataset before adversarial training',
                                                              min_value=1,
                                                              max_value=1000000,
                                                              value=1,
                                                              key='num_clean_epochs')
            if 'attack_epoch_interval' in toolkit['TRAINING_PARAMS']:
                toolkit['attack_epoch_interval'] = st.number_input('Generate a new adversarial training set every '
                                                                   'N epochs',
                                                                   min_value=1,
                                                                   max_value=1000000,
                                                                   value=1,
                                                                   key='attack_epoch_interval')
            if 'early_stopping_epochs' in toolkit['TRAINING_PARAMS']:
                toolkit['early_stopping_epochs'] = st.number_input('Number of epochs validation must increase '
                                                                   'before stopping early',
                                                                   min_value=1,
                                                                   max_value=1000000,
                                                                   value=1,
                                                                   key='early_stopping_epochs')
            else:
                toolkit['early_stopping_epochs'] = None
            if 'learning_rate' in toolkit['TRAINING_PARAMS']:
                toolkit['learning_rate'] = st.number_input('Number of epochs validation must increase before '
                                                           'stopping early',
                                                           min_value=0,
                                                           max_value=1,
                                                           value=5e-5,
                                                           step=0.000001,
                                                           format='.%6f',
                                                           key='learning_rate')
            if 'num_warmup_steps' in toolkit['TRAINING_PARAMS']:
                if st.checkbox('Define in float?'):
                    toolkit['num_warmup_steps'] = st.number_input('The number of steps for the warmup phase of '
                                                                  'linear scheduler',
                                                                  min_value=0,
                                                                  max_value=1,
                                                                  value=0.50,
                                                                  step=0.001,
                                                                  format='.%3f',
                                                                  key='num_warmup_steps')
                else:
                    toolkit['num_warmup_steps'] = st.number_input('The number of steps for the warmup phase of '
                                                                  'linear scheduler',
                                                                  min_value=1,
                                                                  max_value=1000000,
                                                                  value=500,
                                                                  key='num_warmup_steps')
            if 'weight_decay' in toolkit['TRAINING_PARAMS']:
                toolkit['weight_decay'] = st.number_input('Weight decay (L2 penalty)',
                                                          min_value=0,
                                                          max_value=1,
                                                          value=0.01,
                                                          step=0.01,
                                                          format='.%2f',
                                                          key='weight_decay')
            if 'per_device_train_batch_size' in toolkit['TRAINING_PARAMS']:
                toolkit['per_device_train_batch_size'] = st.number_input('The batch size per GPU/CPU for training',
                                                                         min_value=1,
                                                                         max_value=1000000,
                                                                         value=8,
                                                                         key='per_device_train_batch_size')
            if 'per_device_eval_batch_size' in toolkit['TRAINING_PARAMS']:
                toolkit['per_device_eval_batch_size'] = st.number_input('The batch size per GPU/CPU for evaluation',
                                                                        min_value=1,
                                                                        max_value=1000000,
                                                                        value=32,
                                                                        key='per_device_eval_batch_size')
            if 'gradient_accumulation_steps' in toolkit['TRAINING_PARAMS']:
                toolkit['gradient_accumulation_steps'] = st.number_input('Number of updates steps to accumulate '
                                                                         'the gradients before performing a '
                                                                         'backward/update pass',
                                                                         min_value=1,
                                                                         max_value=1000000,
                                                                         value=32,
                                                                         key='gradient_accumulation_steps')
            if 'random_seed' in toolkit['TRAINING_PARAMS']:
                toolkit['random_seed'] = st.number_input('Random seed for reproducibility',
                                                         min_value=1,
                                                         max_value=1000000,
                                                         value=32,
                                                         key='random_seed')
            if 'parallel' in toolkit['TRAINING_PARAMS']:
                toolkit['parallel'] = st.checkbox('Use Multiple GPUs using torch.DataParallel class?',
                                                  value=False,
                                                  key='parallel')
            if 'load_best_model_at_end' in toolkit['TRAINING_PARAMS']:
                toolkit['load_best_model_at_end'] = st.checkbox('keep track of the best model across training and '
                                                                'load it at the end',
                                                                value=False,
                                                                key='parallel')
            if 'alpha' in toolkit['TRAINING_PARAMS']:
                toolkit['alpha'] = st.number_input('The weight for adversarial loss',
                                                   min_value=0,
                                                   max_value=1,
                                                   value=0.50,
                                                   step=0.001,
                                                   format='.%3f',
                                                   key='alpha')
            if 'num_train_adv_examples' in toolkit['TRAINING_PARAMS']:
                if st.checkbox('Use Float Parameters?'):
                    toolkit['num_train_adv_examples'] = st.number_input('The number of samples to successfully '
                                                                        'attack when generating adversarial '
                                                                        'training set before start of every epoch',
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        value=0.50,
                                                                        step=0.001,
                                                                        format='.%3f',
                                                                        key='num_train_adv_examples')
                else:
                    toolkit['num_train_adv_examples'] = st.number_input('The number of samples to successfully '
                                                                        'attack when generating adversarial '
                                                                        'training set before start of every epoch',
                                                                        min_value=1,
                                                                        max_value=1000000,
                                                                        value=8,
                                                                        key='per_device_train_batch_size')
            if 'query_budget_train' in toolkit['TRAINING_PARAMS']:
                if st.checkbox('Set Max Query Budget?', value=False):
                    toolkit['query_budget_train'] = st.number_input('The max query budget to use when generating '
                                                                    'adversarial training set',
                                                                    min_value=1,
                                                                    max_value=1000000,
                                                                    value=1,
                                                                    key='query_budget_train')
                else:
                    toolkit['query_budget_train'] = None
            if 'attack_num_workers_per_device' in toolkit['TRAINING_PARAMS']:
                if st.checkbox('Set Number of Worker Process to run attack?', value=False):
                    toolkit['attack_num_workers_per_device'] = st.number_input('Number of worker processes to run '
                                                                               'per device for attack',
                                                                               min_value=1,
                                                                               max_value=1000000,
                                                                               value=1,
                                                                               key='attack_num_workers_per_device')
                else:
                    toolkit['attack_num_workers_per_device'] = 1
            if 'output_dir' in toolkit['TRAINING_PARAMS']:
                dt = datetime.now()
                toolkit['output_dir'] = st.text_input('Directory to output training logs and checkpoints',
                                                      value=f'/outputs/{dt.strftime("%Y-%m-%d-%H-%M-%S-%f")}',
                                                      key='output_dir')
            if 'checkpoint_interval_steps' in toolkit['TRAINING_PARAMS']:
                if st.checkbox('Save Model Checkpoint after every N updates?'):
                    toolkit['checkpoint_interval_steps'] = st.number_input('Save after N updates',
                                                                           min_value=1,
                                                                           max_value=1000000,
                                                                           value=1,
                                                                           key='checkpoint_interval_steps')
                else:
                    toolkit['checkpoint_interval_steps'] = None
            if 'checkpoint_interval_epochs' in toolkit['TRAINING_PARAMS']:
                if st.checkbox('Save Model Checkpoint after every N epochs?'):
                    toolkit['checkpoint_interval_epochs'] = st.number_input('Save after N epochs',
                                                                            min_value=1,
                                                                            max_value=1000000,
                                                                            value=1,
                                                                            key='checkpoint_interval_epochs')
                else:
                    toolkit['checkpoint_interval_epochs'] = None
            if 'save_last' in toolkit['TRAINING_PARAMS']:
                toolkit['save_last'] = st.checkbox('Save the model at end of training',
                                                   value=True,
                                                   key='save_last')
            if 'log_to_tb' in toolkit['TRAINING_PARAMS']:
                toolkit['log_to_tb'] = st.checkbox('Log to Tensorboard',
                                                   value=False,
                                                   key='log_to_tb')
            if 'tb_log_dir' in toolkit['TRAINING_PARAMS']:
                toolkit['tb_log_dir'] = st.text_input('Directory to output training logs and checkpoints',
                                                      value=r'./runs',
                                                      key='tb_log_dir')
            if 'log_to_wandb' in toolkit['TRAINING_PARAMS']:
                toolkit['log_to_wandb'] = st.checkbox('Log to Wandb',
                                                      value=False,
                                                      key='log_to_wandb')
            if 'wandb_project' in toolkit['TRAINING_PARAMS']:
                toolkit['wandb_project'] = st.text_input('Name of Wandb project for logging',
                                                         value=r'textattack',
                                                         key='wandb_project')
            if 'logging_interval_step' in toolkit['TRAINING_PARAMS']:
                toolkit['logging_interval_step'] = st.number_input('Log to Tensorboard/Wandb every N training '
                                                                   'steps',
                                                                   min_value=1,
                                                                   max_value=1000000,
                                                                   value=1,
                                                                   key='logging_interval_step')

        st.markdown('### Model and Data Selection')
        toolkit['MODEL'] = st.selectbox('Choose Model to Use',
                                        toolkit['ML_POSSIBLE_PICKS'])
        toolkit['DATASET'] = st.selectbox('Choose Dataset to Use',
                                          toolkit['DATASET_POSSIBLE_PICKS'],
                                          help='Due to the sheer number of datasets availble on HuggingFace, '
                                               'we have only provided the top 100 datasets on the website. If you '
                                               'wish to use another dataset not specified here, choose OTHERS.')

        if toolkit['DATASET'] == 'OTHERS':
            toolkit['DATASET'] = st.text_input('Key in the dataset name you wish to use from HuggingFace', key='ds')
        toolkit['TASK_TYPE'] = st.selectbox('Choose Task for Model to Complete', ('classification', 'regression'))

        if len(toolkit['SUBSET_MAPPINGS'][toolkit['DATASET']][0]) != 0:
            toolkit['SUBSET'] = st.selectbox('Select Subset of Data to Use',
                                             toolkit['SUBSET_MAPPINGS'][toolkit['DATASET']][0])
        else:
            toolkit['SUBSET'] = None

        if len(toolkit['SUBSET_MAPPINGS'][toolkit['DATASET']][1]) != 0:
            toolkit['MODEL_COL'] = st.selectbox('Select Data Columns to Use',
                                                toolkit['SUBSET_MAPPINGS'][toolkit['DATASET']][1],
                                                key='column_dat')
        else:
            toolkit['MODEL_COL'] = None

        if len(toolkit['SUBSET_MAPPINGS'][toolkit['DATASET']][2]) > 0:
            toolkit['SPLIT_TRAIN'] = st.selectbox('Select Training Split to Use',
                                                  toolkit['SUBSET_MAPPINGS'][toolkit['DATASET']][2],
                                                  key='train')
            toolkit['SPLIT_TEST'] = st.selectbox('Select Testing Split to Use',
                                                 toolkit['SUBSET_MAPPINGS'][toolkit['DATASET']][2],
                                                 key='test')
            if toolkit['SPLIT_TRAIN'] == toolkit['SPLIT_TEST']:
                st.warning('**Warning**: Your Training and Testing Dataset should not be the same. Ensure that '
                           'you have selected the right dataset to use for your model.')
        else:
            st.warning('**Warning:** This dataset does not have data split properly. You may wish to use another '
                       'dataset or to edit the dataset before passing it into the model for training.')
            toolkit['SPLIT_TRAIN'] = None
            toolkit['SPLIT_TEST'] = None

        st.markdown('### Dataset Explorer\n\n'
                    'Use the above flags to define the Dataset to download and explore.')
        st.info(f'**Current Dataset Chosen**: {toolkit["DATASET"]}')
        if st.button(f'Explore {toolkit["DATASET"]}'):
            train = textattack.datasets.HuggingFaceDataset(name_or_dataset=toolkit['DATASET'],
                                                           subset=toolkit['SUBSET'],
                                                           dataset_columns=toolkit['MODEL_COL'],
                                                           split=toolkit['SPLIT_TRAIN'])
            test = textattack.datasets.HuggingFaceDataset(name_or_dataset=toolkit['DATASET'],
                                                          subset=toolkit['SUBSET'],
                                                          dataset_columns=toolkit['MODEL_COL'],
                                                          split=toolkit['SPLIT_TEST'])
            st.markdown(f'### Training Data\n\n'
                        f'**First Entry**: {train[0]}\n\n'
                        f'**Last Entry**: {train[-1]}\n\n'
                        f'**Length of Dataset**: {len(train)}')
            st.markdown(f'### Testing Data\n\n'
                        f'**First Entry**: {test[0]}\n\n'
                        f'**Last Entry**: {test[-1]}\n\n'
                        f'**Length of Dataset**: {len(test)}')

        if st.checkbox('Attack Model with confusion datasets?', value=False):
            toolkit['ATTACK'] = st.selectbox('Choose Attack recipes to execute on Model',
                                             toolkit['ATTACK_RECIPES'])
            if toolkit['ATTACK'] == 'None':
                toolkit['ATTACK_MODEL'] = None

    st.markdown('## Begin Training\n\n'
                'Kindly ensure that the models you have chosen above is compatible with the dataset ')
    if st.button('Proceed'):
        if toolkit['API']:
            toolkit['ML_MODEL'] = transformers.AutoModelForSequenceClassification.from_pretrained(toolkit['MODEL'])
            toolkit['TOKENIZER'] = transformers.AutoTokenizer.from_pretrained(toolkit['MODEL'])
            toolkit['WRAPPED_MODEL'] = textattack.models.wrappers.HuggingFaceModelWrapper(toolkit['ML_MODEL'],
                                                                                          toolkit['TOKENIZER'])
            toolkit['TRAINING_DATA'] = textattack.datasets.HuggingFaceDataset(
                name_or_dataset=toolkit['DATASET'],
                subset=toolkit['SUBSET'],
                dataset_columns=toolkit['MODEL_COL'],
                split=toolkit['SPLIT_TRAIN']
            )
            toolkit['EVAL_DATA'] = textattack.datasets.HuggingFaceDataset(
                name_or_dataset=toolkit['DATASET'],
                subset=toolkit['SUBSET'],
                dataset_columns=toolkit['MODEL_COL'],
                split=toolkit['SPLIT_TEST']
            )

            if toolkit['ATTACK'] != 'None':
                if toolkit['ATTACK'] == 'A2T (A2T: Attack for Adversarial Training Recipe)':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.A2TYoo2021.build(toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'BAE (BAE: BERT-Based Adversarial Examples)':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.BAEGarg2019.build(toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'BERT-Attack':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.BERTAttackLi2020.build(
                        toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'CheckList':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.CheckList2020.build(
                        toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'CLARE Recipe':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.CLARE2020.build(toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'DeepWordBug':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.DeepWordBugGao2018. \
                        build(toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'Faster Alzantot Genetic Algorithm':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.FasterGeneticAlgorithmJia2019. \
                        build(toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'Alzantot Genetic Algorithm':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.GeneticAlgorithmAlzantot2018. \
                        build(toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'HotFlip':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.HotFlipEbrahimi2017. \
                        build(toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'Improved Genetic Algorithm':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.IGAWang2019.build(toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'Input Reduction':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.InputReductionFeng2018. \
                        build(toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'Kuleshov2017':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.Kuleshov2017.build(toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'MORPHEUS2020':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.MorpheusTan2020.build(
                        toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'Pruthi2019: Combating with Robust Word Recognition':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.Pruthi2019.build(toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'Particle Swarm Optimization':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.PSOZang2020.build(toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'PWWS':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.PWWSRen2019.build(toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'Seq2Sick':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.Seq2SickCheng2018BlackBox. \
                        build(toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'TextBugger':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.TextBuggerLi2018.build(toolkit['WRAPPED_MODEL'])
                elif toolkit['ATTACK'] == 'TextFooler (Is BERT Really Robust?)':
                    toolkit['ATTACK_MODEL'] = textattack.attack_recipes.TextFoolerJin2019. \
                        build(toolkit['WRAPPED_MODEL'])

            toolkit['TRAINING_ARGS'] = textattack.TrainingArgs(
                num_epochs=toolkit['num_epochs'],
                num_clean_epochs=toolkit['num_clean_epochs'],
                attack_epoch_interval=toolkit['attack_epoch_interval'],
                early_stopping_epochs=toolkit['early_stopping_epochs'],
                learning_rate=toolkit['learning_rate'],
                num_warmup_steps=toolkit['num_warmup_steps'],
                weight_decay=toolkit['weight_decay'],
                per_device_train_batch_size=toolkit['per_device_train_batch_size'],
                per_device_eval_batch_size=toolkit['per_device_eval_batch_size'],
                gradient_accumulation_steps=toolkit['gradient_accumulation_steps'],
                random_seed=toolkit['random_seed'],
                parallel=toolkit['parallel'],
                load_best_model_at_end=toolkit['load_best_model_at_end'],
                alpha=toolkit['alpha'],
                num_train_adv_examples=toolkit['num_train_adv_examples'],
                query_budget_train=toolkit['query_budget_train'],
                attack_num_workers_per_device=toolkit['attack_num_workers_per_device'],
                output_dir=toolkit['output_dir'],
                checkpoint_interval_steps=toolkit['checkpoint_interval_steps'],
                checkpoint_interval_epochs=toolkit['checkpoint_interval_epochs'],
                save_last=toolkit['save_last'],
                log_to_tb=toolkit['log_to_tb'],
                tb_log_dir=toolkit['tb_log_dir'],
                log_to_wandb=toolkit['log_to_wandb'],
                wandb_project=toolkit['wandb_project'],
                logging_interval_step=toolkit['logging_interval_step']
            )
            toolkit['TRAINER'] = textattack.Trainer(
                model_wrapper=toolkit['WRAPPED_MODEL'],
                task_type=toolkit['TASK_TYPE'],
                attack=toolkit['ATTACK_MODEL'],
                train_dataset=toolkit['TRAINING_DATA'],
                eval_dataset=toolkit['EVAL_DATA'],
                training_args=toolkit['TRAINING_ARGS']
            )

            with st.spinner('Training Model... Refer to your Terminal for more information...'):
                try:
                    toolkit['TRAINER'].train()
                except Exception as ex:
                    st.error(ex)
                else:
                    st.success(f'Successfully trained model! Model saved in {os.getcwd()}{toolkit["output_dir"]}.')

        else:
            with st.spinner('Training Model... Refer to your Terminal for more information...'):
                try:
                    var_list = ['textattack', 'train',
                                '--model_name_or_path', toolkit['ML_MODEL'],
                                '--attack', toolkit['attack'],
                                '--dataset', toolkit['DATASET'],
                                '--task_type', toolkit['TASK_TYPE'],
                                '--model_max_length', toolkit['model_max_length'],
                                '--model_num_labels', toolkit['model_num_labels'],
                                '--dataset_train_split', toolkit['dataset_train_split'],
                                '--dataset_eval_split', toolkit['dataset_eval_split'],
                                '--filter_train_by_labels', toolkit['filter_train_by_labels'],
                                '--filter_eval_by_labels', toolkit['filter_eval_by_labels'],
                                '--num_epochs', toolkit['num_epochs'],
                                '--num_clean_epochs', toolkit['num_clean_epochs'],
                                '--attack_epoch_interval', toolkit['attack_epoch_interval'],
                                '--early_stopping_epochs', toolkit['early_stopping_epochs'],
                                '--learning_rate', toolkit['learning_rate'],
                                '--num_warmup_steps', toolkit['num_warmup_steps'],
                                '--weight_decay', toolkit['weight_decay'],
                                '--per_device_train_batch_size', toolkit['per_device_train_batch_size'],
                                '--per_device_eval_batch_size', toolkit['per_device_eval_batch_size'],
                                '--gradient_accumulation_steps', toolkit['gradient_accumulation_steps'],
                                '--random_seed', toolkit['random_seed'],
                                '--parallel', toolkit['parallel'],
                                '--load_best_model_at_end', toolkit['load_best_model_at_end'],
                                '--alpha', toolkit['alpha'],
                                '--num_train_adv_examples', toolkit['num_train_adv_examples'],
                                '--query_budget_train', toolkit['query_budget_train'],
                                '--attack_num_workers_per_device', toolkit['attack_num_workers_per_device'],
                                '--output_dir', toolkit['output_dir'],
                                '--checkpoint_interval_steps', toolkit['checkpoint_interval_steps'],
                                '--checkpoint_interval_epochs', toolkit['checkpoint_interval_epochs'],
                                '--save_last', toolkit['save_last'],
                                '--log_to_tb', toolkit['log_to_tb'],
                                '--tb_log_dir', toolkit['tb_log_dir'],
                                '--log_to_wandb', toolkit['log_to_wandb'],
                                '--wandb_project', toolkit['wandb_project'],
                                '--logging_interval_step', toolkit['logging_interval_step']]
                    processor = subprocess.run(var_list)
                except Exception as ex:
                    st.error(ex)
                else:
                    st.success(f'Successfully trained model! Model saved in {os.getcwd()}{toolkit["output_dir"]}.')

    elif toolkit['MODEL_MODE'] == 'Evaluation':
        st.info('This functionality is not implemented yet.')
