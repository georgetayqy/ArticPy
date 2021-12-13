"""
This module allows the user to train models and to predict NLP data
"""

# -------------------------------------------------------------------------------------------------------------------- #
# |                                         IMPORT RELEVANT LIBRARIES                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
import contextlib
import os
import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import textattack.models.wrappers
import torch
import subprocess
import transformers

from datetime import datetime
from io import StringIO
from config import trainer, STREAMLIT_STATIC_PATH, DOWNLOAD_PATH
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import csp_downloaders
from utils.helper import readFile, summarise, modelIterator, printDataFrame, dominantTopic


# -------------------------------------------------------------------------------------------------------------------- #
# |                                             MAIN APP FUNCTIONALITY                                               | #
# -------------------------------------------------------------------------------------------------------------------- #
def app():
    """
    Main function that will be called when the app is run
    """

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
    trainer['MODEL_MODE'] = st.selectbox('Select the actions you want to perform', ('Training', 'Evaluation'))

    if trainer['MODEL_MODE'] == 'Training':
        st.markdown('## Flags\n\n'
                    '### Training Parameters')
        trainer['API'] = st.checkbox('Use Training API?',
                                     help='Note that with this option selected, you must ensure that your GPU has '
                                          'sufficient GPU memory to run the networks/models you selected. If you '
                                          'are unsure, it is better to use the Command Line Argument API to fine '
                                          'tune the model parameters before starting the training.',
                                     value=True)

        if trainer['API']:
            trainer['TRAINING_PARAMS'] = st.multiselect('Select Training Parameters',
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
        else:
            trainer['TRAINING_PARAMS'] = st.multiselect('Select Training Parameters',
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
                                                                 'per_device_train_batch_size', 'model_num_labels'))
            if 'attack' in trainer['TRAINING_PARAMS']:
                trainer['attack'] = st.text_input('Attack string', key='attack')
            else:
                trainer['attack'] = None

            if 'model_max_length' in trainer['TRAINING_PARAMS']:
                if st.checkbox('Define Model Max Length'):
                    trainer['model_max_length'] = st.number_input('Model Max Length',
                                                                  min_value=1,
                                                                  max_value=1000000,
                                                                  value=64,
                                                                  key='model_max_length')
                else:
                    trainer['model_max_length'] = None

            if 'model_num_labels' in trainer['TRAINING_PARAMS']:
                if st.checkbox('Define Number of Labels'):
                    trainer['model_num_labels'] = st.number_input('Number of Labels',
                                                                  min_value=1,
                                                                  max_value=1000000,
                                                                  value=1,
                                                                  key='model_num_labels')
                else:
                    trainer['model_num_labels'] = None

            if 'filter_train_by_labels' in trainer['TRAINING_PARAMS']:
                trainer['filter_train_by_labels'] = st.text_input('Filter Train Data By Labels',
                                                                  key='filter_train')
                trainer['filter_train_by_labels'] = [
                    label for label in trainer['filter_train_by_labels'].split(',')
                ]
            else:
                trainer['filter_train_by_labels'] = None

            if 'filter_eval_by_labels' in trainer['TRAINING_PARAMS']:
                trainer['filter_eval_by_labels'] = st.text_input('Filter Test Data By Labels',
                                                                 key='filter_test')
                trainer['filter_eval_by_labels'] = [
                    label for label in trainer['filter_eval_by_labels'].split(',')
                ]
            else:
                trainer['filter_eval_by_labels'] = None

            if 'num_epochs' in trainer['TRAINING_PARAMS']:
                trainer['num_epochs'] = st.number_input('Total number of epochs for training',
                                                        min_value=1,
                                                        max_value=1000000,
                                                        value=3,
                                                        key='num_epochs')
            else:
                if trainer['API']:
                    trainer['num_epochs'] = 3
                else:
                    trainer['num_epochs'] = None

            if 'num_clean_epochs' in trainer['TRAINING_PARAMS']:
                trainer['num_clean_epochs'] = st.number_input('Number of epochs to train on just the original '
                                                              'training dataset before adversarial training',
                                                              min_value=1,
                                                              max_value=1000000,
                                                              value=1,
                                                              key='num_clean_epochs')
            else:
                if trainer['API']:
                    trainer['num_clean_epochs'] = 1
                else:
                    trainer['num_clean_epochs'] = None

            if 'attack_epoch_interval' in trainer['TRAINING_PARAMS']:
                trainer['attack_epoch_interval'] = st.number_input('Generate a new adversarial training set every '
                                                                   'N epochs',
                                                                   min_value=1,
                                                                   max_value=1000000,
                                                                   value=1,
                                                                   key='attack_epoch_interval')
            else:
                if trainer['API']:
                    trainer['attack_epoch_interval'] = 1
                else:
                    trainer['attack_epoch_interval'] = None

            if 'early_stopping_epochs' in trainer['TRAINING_PARAMS']:
                trainer['early_stopping_epochs'] = st.number_input('Number of epochs validation must increase '
                                                                   'before stopping early',
                                                                   min_value=1,
                                                                   max_value=1000000,
                                                                   value=1,
                                                                   key='early_stopping_epochs')
            else:
                trainer['early_stopping_epochs'] = None

            if 'learning_rate' in trainer['TRAINING_PARAMS']:
                trainer['learning_rate'] = st.number_input('Number of epochs validation must increase before '
                                                           'stopping early',
                                                           min_value=0,
                                                           max_value=1,
                                                           value=5e-5,
                                                           step=0.000001,
                                                           format='.%6f',
                                                           key='learning_rate')
            else:
                if trainer['API']:
                    trainer['learning_rate'] = 5e-5
                else:
                    trainer['learning_rate'] = None

            if 'num_warmup_steps' in trainer['TRAINING_PARAMS']:
                if st.checkbox('Define in float?'):
                    trainer['num_warmup_steps'] = st.number_input('The number of steps for the warmup phase of '
                                                                  'linear scheduler',
                                                                  min_value=0,
                                                                  max_value=1,
                                                                  value=0.50,
                                                                  step=0.001,
                                                                  format='.%3f',
                                                                  key='num_warmup_steps')
                else:
                    trainer['num_warmup_steps'] = st.number_input('The number of steps for the warmup phase of '
                                                                  'linear scheduler',
                                                                  min_value=1,
                                                                  max_value=1000000,
                                                                  value=500,
                                                                  key='num_warmup_steps')
            else:
                if trainer['API']:
                    trainer['num_warmup_steps'] = 500
                else:
                    trainer['num_warmup_steps'] = None

            if 'weight_decay' in trainer['TRAINING_PARAMS']:
                trainer['weight_decay'] = st.number_input('Weight decay (L2 penalty)',
                                                          min_value=0,
                                                          max_value=1,
                                                          value=0.01,
                                                          step=0.01,
                                                          format='.%2f',
                                                          key='weight_decay')
            else:
                if trainer['API']:
                    trainer['weight_decay'] = 0.01
                else:
                    trainer['weight_decay'] = None

            if 'per_device_train_batch_size' in trainer['TRAINING_PARAMS']:
                trainer['per_device_train_batch_size'] = st.number_input('The batch size per GPU/CPU for training',
                                                                         min_value=1,
                                                                         max_value=1000000,
                                                                         value=8,
                                                                         key='per_device_train_batch_size')
            else:
                if trainer['API']:
                    trainer['per_device_train_batch_size'] = 8
                else:
                    trainer['per_device_train_batch_size'] = None

            if 'per_device_eval_batch_size' in trainer['TRAINING_PARAMS']:
                trainer['per_device_eval_batch_size'] = st.number_input('The batch size per GPU/CPU for evaluation',
                                                                        min_value=1,
                                                                        max_value=1000000,
                                                                        value=32,
                                                                        key='per_device_eval_batch_size')
            else:
                if trainer['API']:
                    trainer['per_device_eval_batch_size'] = 32
                else:
                    trainer['per_device_eval_batch_size'] = None

            if 'gradient_accumulation_steps' in trainer['TRAINING_PARAMS']:
                trainer['gradient_accumulation_steps'] = st.number_input('Number of updates steps to accumulate '
                                                                         'the gradients before performing a '
                                                                         'backward/update pass',
                                                                         min_value=1,
                                                                         max_value=1000000,
                                                                         value=32,
                                                                         key='gradient_accumulation_steps')
            else:
                if trainer['API']:
                    trainer['gradient_accumulation_steps'] = 1
                else:
                    trainer['gradient_accumulation_steps'] = None

            if 'random_seed' in trainer['TRAINING_PARAMS']:
                trainer['random_seed'] = st.number_input('Random seed for reproducibility',
                                                         min_value=1,
                                                         max_value=1000000,
                                                         value=786,
                                                         key='random_seed')
            else:
                if trainer['API']:
                    trainer['random_seed'] = 786
                else:
                    trainer['random_seed'] = None

            if 'parallel' in trainer['TRAINING_PARAMS']:
                trainer['parallel'] = st.checkbox('Use Multiple GPUs using torch.DataParallel class?',
                                                  value=False,
                                                  key='parallel')
            else:
                if trainer['API']:
                    trainer['parallel'] = False
                else:
                    trainer['parallel'] = None

            if 'load_best_model_at_end' in trainer['TRAINING_PARAMS']:
                trainer['load_best_model_at_end'] = st.checkbox('keep track of the best model across training and '
                                                                'load it at the end',
                                                                value=False,
                                                                key='parallel')
            else:
                trainer['load_best_model_at_end'] = False

            if 'alpha' in trainer['TRAINING_PARAMS']:
                trainer['alpha'] = st.number_input('The weight for adversarial loss',
                                                   min_value=0,
                                                   max_value=1,
                                                   value=0.50,
                                                   step=0.001,
                                                   format='.%3f',
                                                   key='alpha')
            else:
                if trainer['API']:
                    trainer['alpha'] = 1.0
                else:
                    trainer['alpha'] = None

            if 'num_train_adv_examples' in trainer['TRAINING_PARAMS']:
                if st.checkbox('Use Float Parameters?'):
                    trainer['num_train_adv_examples'] = st.number_input('The number of samples to successfully '
                                                                        'attack when generating adversarial '
                                                                        'training set before start of every epoch',
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        value=0.50,
                                                                        step=0.001,
                                                                        format='.%3f',
                                                                        key='num_train_adv_examples')
                else:
                    trainer['num_train_adv_examples'] = st.number_input('The number of samples to successfully '
                                                                        'attack when generating adversarial '
                                                                        'training set before start of every epoch',
                                                                        min_value=1,
                                                                        max_value=1000000,
                                                                        value=8,
                                                                        key='per_device_train_batch_size')
            else:
                if trainer['API']:
                    trainer['num_train_adv_examples'] = -1
                else:
                    trainer['num_train_adv_examples'] = None

            if 'query_budget_train' in trainer['TRAINING_PARAMS']:
                if st.checkbox('Set Max Query Budget?', value=False):
                    trainer['query_budget_train'] = st.number_input('The max query budget to use when generating '
                                                                    'adversarial training set',
                                                                    min_value=1,
                                                                    max_value=1000000,
                                                                    value=1,
                                                                    key='query_budget_train')
                else:
                    trainer['query_budget_train'] = None

            if 'attack_num_workers_per_device' in trainer['TRAINING_PARAMS']:
                if st.checkbox('Set Number of Worker Process to run attack?', value=False):
                    trainer['attack_num_workers_per_device'] = st.number_input('Number of worker processes to run '
                                                                               'per device for attack',
                                                                               min_value=1,
                                                                               max_value=1000000,
                                                                               value=1,
                                                                               key='attack_num_workers_per_device')
                else:
                    if trainer['API']:
                        trainer['attack_num_workers_per_device'] = 1
                    else:
                        trainer['attack_num_workers_per_device'] = None

            if 'output_dir' in trainer['TRAINING_PARAMS']:
                dt = datetime.now()
                trainer['output_dir'] = st.text_input('Directory to output training logs and checkpoints',
                                                      value=f'/outputs/{dt.strftime("%Y-%m-%d-%H-%M-%S-%f")}',
                                                      key='output_dir')
            else:
                trainer['output_dir'] = f'/outputs/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")}'

            if 'checkpoint_interval_steps' in trainer['TRAINING_PARAMS']:
                if st.checkbox('Save Model Checkpoint after every N updates?'):
                    trainer['checkpoint_interval_steps'] = st.number_input('Save after N updates',
                                                                           min_value=1,
                                                                           max_value=1000000,
                                                                           value=1,
                                                                           key='checkpoint_interval_steps')
                else:
                    trainer['checkpoint_interval_steps'] = None

            if 'checkpoint_interval_epochs' in trainer['TRAINING_PARAMS']:
                if st.checkbox('Save Model Checkpoint after every N epochs?'):
                    trainer['checkpoint_interval_epochs'] = st.number_input('Save after N epochs',
                                                                            min_value=1,
                                                                            max_value=1000000,
                                                                            value=1,
                                                                            key='checkpoint_interval_epochs')
                else:
                    trainer['checkpoint_interval_epochs'] = None

            if 'save_last' in trainer['TRAINING_PARAMS']:
                trainer['save_last'] = st.checkbox('Save the model at end of training',
                                                   value=True,
                                                   key='save_last')
            else:
                if trainer['API']:
                    trainer['save_last'] = True
                else:
                    trainer['save_last'] = None

            if 'log_to_tb' in trainer['TRAINING_PARAMS']:
                trainer['log_to_tb'] = st.checkbox('Log to Tensorboard',
                                                   value=False,
                                                   key='log_to_tb')
            else:
                if trainer['API']:
                    trainer['log_to_tb'] = False
                else:
                    trainer['log_to_tb'] = None

            if 'tb_log_dir' in trainer['TRAINING_PARAMS']:
                trainer['tb_log_dir'] = st.text_input('Directory to output training logs and checkpoints',
                                                      value=r'./runs',
                                                      key='tb_log_dir')
            else:
                trainer['tb_log_dir'] = r'./runs'

            if 'log_to_wandb' in trainer['TRAINING_PARAMS']:
                trainer['log_to_wandb'] = st.checkbox('Log to Wandb',
                                                      value=False,
                                                      key='log_to_wandb')
            else:
                if trainer['API']:
                    trainer['log_to_wandb'] = False
                else:
                    trainer['log_to_wandb'] = None

            if 'wandb_project' in trainer['TRAINING_PARAMS']:
                trainer['wandb_project'] = st.text_input('Name of Wandb project for logging',
                                                         value=r'textattack',
                                                         key='wandb_project')
            else:
                if trainer['API']:
                    trainer['wandb_project'] = 'textattack'
                else:
                    trainer['wandb_project'] = None

            if 'logging_interval_step' in trainer['TRAINING_PARAMS']:
                trainer['logging_interval_step'] = st.number_input('Log to Tensorboard/Wandb every N training '
                                                                   'steps',
                                                                   min_value=1,
                                                                   max_value=1000000,
                                                                   value=1,
                                                                   key='logging_interval_step')
            else:
                if trainer['API']:
                    trainer['logging_interval_step'] = 1
                else:
                    trainer['logging_interval_step'] = None

        st.markdown('### Model and Data Selection')
        trainer['MODEL'] = st.selectbox('Choose Model to Use',
                                        trainer['ML_POSSIBLE_PICKS'])
        trainer['DATASET'] = st.selectbox('Choose Dataset to Use',
                                          trainer['DATASET_POSSIBLE_PICKS'],
                                          help='Due to the sheer number of datasets availble on HuggingFace, '
                                               'we have only provided the top 100 datasets on the website. If you '
                                               'wish to use another dataset not specified here, choose OTHERS.')

        if trainer['DATASET'] == 'OTHERS':
            trainer['DATASET'] = st.text_input('Key in the dataset name you wish to use from HuggingFace', key='ds')
        trainer['TASK_TYPE'] = st.selectbox('Choose Task for Model to Complete', ('classification', 'regression'))

        if len(trainer['SUBSET_MAPPINGS'][trainer['DATASET']][0]) != 0:
            trainer['SUBSET'] = st.selectbox('Select Subset of Data to Use',
                                             trainer['SUBSET_MAPPINGS'][trainer['DATASET']][0])
        else:
            trainer['SUBSET'] = None

        if len(trainer['SUBSET_MAPPINGS'][trainer['DATASET']][1]) != 0:
            trainer['MODEL_COL'] = st.selectbox('Select Data Columns to Use',
                                                trainer['SUBSET_MAPPINGS'][trainer['DATASET']][1],
                                                key='column_dat')
        else:
            trainer['MODEL_COL'] = None

        if len(trainer['SUBSET_MAPPINGS'][trainer['DATASET']][2]) > 0:
            trainer['SPLIT_TRAIN'] = st.selectbox('Select Training Split to Use',
                                                  trainer['SUBSET_MAPPINGS'][trainer['DATASET']][2],
                                                  key='train')
            trainer['SPLIT_TEST'] = st.selectbox('Select Testing Split to Use',
                                                 trainer['SUBSET_MAPPINGS'][trainer['DATASET']][2],
                                                 key='test')
            if trainer['SPLIT_TRAIN'] == trainer['SPLIT_TEST']:
                st.warning('**Warning**: Your Training and Testing Dataset should not be the same. Ensure that '
                           'you have selected the right dataset to use for your model.')
        else:
            st.warning('**Warning:** This dataset does not have data split properly. You may wish to use another '
                       'dataset or to edit the dataset before passing it into the model for training.')
            trainer['SPLIT_TRAIN'] = None
            trainer['SPLIT_TEST'] = None

        st.markdown('### Dataset Explorer\n\n'
                    'Use the above flags to define the Dataset to download and explore.')
        st.info(f'**Current Dataset Chosen**: {trainer["DATASET"]}')
        if st.button(f'Explore {trainer["DATASET"]}'):
            train = textattack.datasets.HuggingFaceDataset(name_or_dataset=trainer['DATASET'],
                                                           subset=trainer['SUBSET'],
                                                           dataset_columns=trainer['MODEL_COL'],
                                                           split=trainer['SPLIT_TRAIN'])
            test = textattack.datasets.HuggingFaceDataset(name_or_dataset=trainer['DATASET'],
                                                          subset=trainer['SUBSET'],
                                                          dataset_columns=trainer['MODEL_COL'],
                                                          split=trainer['SPLIT_TEST'])
            st.markdown(f'### Training Data\n\n'
                        f'**First Entry**: {train[0]}\n\n'
                        f'**Last Entry**: {train[-1]}\n\n'
                        f'**Length of Dataset**: {len(train)}')
            st.markdown(f'### Testing Data\n\n'
                        f'**First Entry**: {test[0]}\n\n'
                        f'**Last Entry**: {test[-1]}\n\n'
                        f'**Length of Dataset**: {len(test)}')

        if st.checkbox('Attack Model with confusion datasets?', value=False):
            trainer['ATTACK'] = st.selectbox('Choose Attack recipes to execute on Model',
                                             trainer['ATTACK_RECIPES'])
            if trainer['ATTACK'] == 'None':
                trainer['ATTACK_MODEL'] = None

    st.markdown('## Begin Training\n\n'
                'Kindly ensure that the models you have chosen above is compatible with the dataset ')
    if st.button('Proceed'):
        if trainer['API']:
            trainer['ML_MODEL'] = transformers.AutoModelForSequenceClassification.from_pretrained(trainer['MODEL'])
            trainer['TOKENIZER'] = transformers.AutoTokenizer.from_pretrained(trainer['MODEL'])
            trainer['WRAPPED_MODEL'] = textattack.models.wrappers.HuggingFaceModelWrapper(trainer['ML_MODEL'],
                                                                                          trainer['TOKENIZER'])
            trainer['TRAINING_DATA'] = textattack.datasets.HuggingFaceDataset(
                name_or_dataset=trainer['DATASET'],
                subset=trainer['SUBSET'],
                dataset_columns=trainer['MODEL_COL'],
                split=trainer['SPLIT_TRAIN']
            )
            trainer['EVAL_DATA'] = textattack.datasets.HuggingFaceDataset(
                name_or_dataset=trainer['DATASET'],
                subset=trainer['SUBSET'],
                dataset_columns=trainer['MODEL_COL'],
                split=trainer['SPLIT_TEST']
            )

            if trainer['ATTACK'] != 'None':
                if trainer['ATTACK'] == 'A2T (A2T: Attack for Adversarial Training Recipe)':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.A2TYoo2021.build(trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'BAE (BAE: BERT-Based Adversarial Examples)':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.BAEGarg2019.build(trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'BERT-Attack':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.BERTAttackLi2020.build(
                        trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'CheckList':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.CheckList2020.build(
                        trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'CLARE Recipe':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.CLARE2020.build(trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'DeepWordBug':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.DeepWordBugGao2018. \
                        build(trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'Faster Alzantot Genetic Algorithm':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.FasterGeneticAlgorithmJia2019. \
                        build(trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'Alzantot Genetic Algorithm':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.GeneticAlgorithmAlzantot2018. \
                        build(trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'HotFlip':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.HotFlipEbrahimi2017. \
                        build(trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'Improved Genetic Algorithm':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.IGAWang2019.build(trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'Input Reduction':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.InputReductionFeng2018. \
                        build(trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'Kuleshov2017':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.Kuleshov2017.build(trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'MORPHEUS2020':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.MorpheusTan2020.build(
                        trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'Pruthi2019: Combating with Robust Word Recognition':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.Pruthi2019.build(trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'Particle Swarm Optimization':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.PSOZang2020.build(trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'PWWS':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.PWWSRen2019.build(trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'Seq2Sick':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.Seq2SickCheng2018BlackBox. \
                        build(trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'TextBugger':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.TextBuggerLi2018.build(trainer['WRAPPED_MODEL'])
                elif trainer['ATTACK'] == 'TextFooler (Is BERT Really Robust?)':
                    trainer['ATTACK_MODEL'] = textattack.attack_recipes.TextFoolerJin2019. \
                        build(trainer['WRAPPED_MODEL'])

            trainer['TRAINING_ARGS'] = textattack.TrainingArgs(
                num_epochs=trainer['num_epochs'],
                num_clean_epochs=trainer['num_clean_epochs'],
                attack_epoch_interval=trainer['attack_epoch_interval'],
                early_stopping_epochs=trainer['early_stopping_epochs'],
                learning_rate=trainer['learning_rate'],
                num_warmup_steps=trainer['num_warmup_steps'],
                weight_decay=trainer['weight_decay'],
                per_device_train_batch_size=trainer['per_device_train_batch_size'],
                per_device_eval_batch_size=trainer['per_device_eval_batch_size'],
                gradient_accumulation_steps=trainer['gradient_accumulation_steps'],
                random_seed=trainer['random_seed'],
                parallel=trainer['parallel'],
                load_best_model_at_end=trainer['load_best_model_at_end'],
                alpha=trainer['alpha'],
                num_train_adv_examples=trainer['num_train_adv_examples'],
                query_budget_train=trainer['query_budget_train'],
                attack_num_workers_per_device=trainer['attack_num_workers_per_device'],
                output_dir=trainer['output_dir'],
                checkpoint_interval_steps=trainer['checkpoint_interval_steps'],
                checkpoint_interval_epochs=trainer['checkpoint_interval_epochs'],
                save_last=trainer['save_last'],
                log_to_tb=trainer['log_to_tb'],
                tb_log_dir=trainer['tb_log_dir'],
                log_to_wandb=trainer['log_to_wandb'],
                wandb_project=trainer['wandb_project'],
                logging_interval_step=trainer['logging_interval_step']
            )
            trainer['TRAINER'] = textattack.Trainer(
                model_wrapper=trainer['WRAPPED_MODEL'],
                task_type=trainer['TASK_TYPE'],
                attack=trainer['ATTACK_MODEL'],
                train_dataset=trainer['TRAINING_DATA'],
                eval_dataset=trainer['EVAL_DATA'],
                training_args=trainer['TRAINING_ARGS']
            )

            with st.spinner('Training Model... Refer to your Terminal for more information...'):
                try:
                    trainer['TRAINER'].train()
                except Exception as ex:
                    st.error(ex)
                else:
                    st.success(f'Successfully trained model! Model saved in {os.getcwd()}{trainer["output_dir"]}.')

        else:
            with st.spinner('Training Model... Refer to your Terminal for more information...'):
                var_list = ['textattack', 'train']
                maps = {
                    'model_name_or_path': ['--model-name-or-path', trainer['MODEL']],
                    'dataset': ['--dataset', trainer['DATASET']],
                    'attack': ['--attack', trainer['attack']],
                    'task_type': ['--task-type', trainer['TASK_TYPE']],
                    'model_max_length': ['--model-max-length', trainer['model_max_length']],
                    'model_num_labels': ['--model-num-labels', trainer['model_num_labels']],
                    'dataset_train_split': ['--dataset-train-split', trainer['dataset_train_split']],
                    'dataset_eval_split': ['--dataset-eval-split', trainer['dataset_eval_split']],
                    'filter_train_by_labels': ['--filter-train-by-labels', trainer['filter_train_by_labels']],
                    'filter_eval_by_labels': ['--filter-eval-by-labels', trainer['filter_eval_by_labels']],
                    'num_epochs': ['--num-epochs', trainer['num_epochs']],
                    'num_clean_epochs': ['--num-clean-epochs', trainer['num_clean_epochs']],
                    'attack_epoch_interval': ['--attack-epoch-interval', trainer['attack_epoch_interval']],
                    'early_stopping_epochs': ['--early-stopping-epochs', trainer['early_stopping_epochs']],
                    'learning_rate': ['--learning-rate', trainer['learning_rate']],
                    'num_warmup_steps': ['--num-warmup-steps', trainer['num_warmup_steps']],
                    'weight_decay': ['--weight-decay', trainer['weight_decay']],
                    'per_device_train_batch_size': ['--per-device-train-batch-size',
                                                    trainer['per_device_train_batch_size']],
                    'per_device_eval_batch_size': ['--per-device-eval-batch-size',
                                                   trainer['per_device_eval_batch_size']],
                    'gradient_accumulation_steps': ['--gradient-accumulation-steps',
                                                    trainer['gradient_accumulation_steps']],
                    'random_seed': ['--random-seed', trainer['random_seed']],
                    'parallel': ['--parallel', trainer['parallel']],
                    'load_best_model_at_end': ['--load-best-model-at-end', trainer['load_best_model_at_end']],
                    'alpha': ['--alpha', trainer['alpha']],
                    'num_train_adv_examples': ['--num-train-adv-examples', trainer['num_train_adv_examples']],
                    'query_budget_train': ['--query-budget-train', trainer['query_budget_train']],
                    'attack_num_workers_per_device': ['--attack-num-workers-per-device',
                                                      trainer['attack_num_workers_per_device']],
                    'output_dir': ['--output-dir', trainer['output_dir']],
                    'checkpoint_interval_steps': ['--checkpoint-interval-steps',
                                                  trainer['checkpoint_interval_steps']],
                    'checkpoint_interval_epochs': ['--checkpoint-interval-epochs',
                                                   trainer['checkpoint_interval_epochs']],
                    'save_last': ['--save-last', trainer['save_last']],
                    'log_to_tb': ['--log-to-tb', trainer['log_to_tb']],
                    'tb_log_dir': ['--tb-log-dir', trainer['tb_log_dir']],
                    'log_to_wandb': ['--log-to-wandb', trainer['log_to_wandb']],
                    'wandb_project': ['--wandb-project', trainer['wandb_project']],
                    'logging_interval_step': ['--logging-interval-step',
                                              trainer['logging_interval_step']]
                }

                # only include variables that are defined
                bools = [None, True, False]
                maps = {key: value for key, value in maps.items() if value[1] not in bools}
                for k, v in maps.items():
                    var_list.extend(v)

                var_list = [str(iter_) for iter_ in var_list]

                # run the command
                # code taken from https://gist.github.com/andfanilo/aa3e4a6a15124c58e88262e193e1febf
                st.markdown('### Outputs')
                try:
                    processor = subprocess.run(var_list)
                except Exception as ex:
                    st.error(ex)
                else:
                    st.success(f'Successfully trained model! Model saved in {os.getcwd()}{trainer["output_dir"]}.')

    elif trainer['MODEL_MODE'] == 'Evaluation':
        st.markdown('## Flags')
        trainer['SAVE'] = st.checkbox('Save Outputs?', help='Due to the possibility of files with the same file name '
                                                            'and content being downloaded again, a unique file '
                                                            'identifier is tacked onto the filename.')
        trainer['VERBOSE'] = st.checkbox('Display Outputs?')

        if trainer['VERBOSE']:
            trainer['VERBOSITY'] = st.slider('Data points',
                                             key='Data points to display?',
                                             min_value=0,
                                             max_value=1000,
                                             value=20,
                                             help='Select 0 to display all Data Points')
            trainer['ADVANCED_ANALYSIS'] = st.checkbox('Display Advanced DataFrame Statistics?',
                                                       help='This option will analyse your DataFrame and display '
                                                            'advanced statistics on it. Note that this will require '
                                                            'some time and processing power to complete. Deselect this '
                                                            'option if this if you do not require it.')
        trainer['PRED_FILE'] = st.checkbox('Load Predictions from File?', key='preds')

        if trainer['PRED_FILE']:
            st.markdown('## Upload Data\n'
                        'Due to limitations imposed by the file uploader widget, only files smaller than 200 MB can be '
                        'loaded with the widget. To circumvent this limitation, you may choose to '
                        'rerun the app with the tag `--server.maxUploadSize=[SIZE_IN_MB_HERE]` appended behind the '
                        '`streamlit run app.py` command and define the maximum size of file you can upload '
                        'onto Streamlit (replace `SIZE_IN_MB_HERE` with an integer value above). Do note that this '
                        'option is only available for users who run the app using the app\'s source code or through '
                        'Docker. For Docker, you will need to append the tag above behind the Docker Image name when '
                        'running the `run` command, e.g. '
                        '`docker run asdfghjklxl/news:latest --server.maxUploadSize=1028`; if you do '
                        'not use the tag, the app will run with a default maximum upload size of 200 MB.\n\n'
                        'Alternatively, you may use the Large File option to pull your dataset from any one of the '
                        'four supported Cloud Service Providers into the app.\n\n'
                        'After selecting the size of your file, select the file format you wish to upload.\n\n')
            trainer['FILE'] = st.selectbox('Select the Size of File to Load', ('Small File(s)', 'Large File(s)'))
            trainer['MODE'] = st.selectbox('Define the Data Input Format', ('CSV', 'XLSX'))

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                 FILE UPLOADING                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
            if trainer['FILE'] == 'Small File(s)':
                st.markdown('### Upload File\n')
                trainer['PRED_FILEPATH'] = st.file_uploader(f'Load {trainer["MODE"]} File', type=[trainer['MODE']])
                if trainer['PRED_FILEPATH'] is not None:
                    trainer['PRED_DATA'] = readFile(trainer['PRED_FILEPATH'], trainer['MODE'])
                    if not trainer['PRED_DATA'].empty:
                        trainer['PRED_DATA'] = trainer['PRED_DATA'].astype(str)
                        trainer['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored',
                                                              list(trainer['PRED_DATA'].columns))
                        st.success(f'Data Loaded from {trainer["DATA_COLUMN"]}!')
                else:
                    trainer['PRED_DATA'] = pd.DataFrame()

            elif trainer['FILE'] == 'Large File(s)':
                st.info(f'File Format Selected: **{trainer["MODE"]}**')
                trainer['CSP'] = st.selectbox('CSP', ('Select a CSP', 'Azure', 'Amazon', 'Google'))

                if trainer['CSP'] == 'Azure':
                    azure = csp_downloaders.AzureDownloader()
                    if azure.SUCCESSFUL:
                        try:
                            azure.downloadBlob()
                            trainer['PRED_DATA'] = readFile(azure.AZURE_DOWNLOAD_PATH, trainer['MODE'])
                        except Exception as ex:
                            st.error(f'Error: {ex}. Try again.')

                    if not trainer['PRED_DATA'].empty:
                        trainer['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored',
                                                              list(trainer['PRED_DATA'].columns))
                        st.success(f'Data Loaded from {trainer["DATA_COLUMN"]}!')

                elif trainer['CSP'] == 'Amazon':
                    aws = csp_downloaders.AWSDownloader()
                    if aws.SUCCESSFUL:
                        try:
                            aws.downloadFile()
                            trainer['PRED_DATA'] = readFile(aws.AWS_FILE_NAME, trainer['MODE'])
                        except Exception as ex:
                            st.error(f'Error: {ex}. Try again.')

                    if not trainer['PRED_DATA'].empty:
                        trainer['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored',
                                                              list(trainer['PRED_DATA'].columns))
                        st.success(f'Data Loaded from {trainer["DATA_COLUMN"]}!')

                elif trainer['CSP'] == 'Google':
                    gcs = csp_downloaders.GoogleDownloader()
                    if gcs.SUCCESSFUL:
                        try:
                            gcs.downloadBlob()
                            trainer['PRED_DATA'] = readFile(gcs.GOOGLE_DESTINATION_FILE_NAME, trainer['MODE'])
                        except Exception as ex:
                            st.error(f'Error: {ex}. Try again.')

                    if not trainer['PRED_DATA'].empty:
                        trainer['DATA_COLUMN'] = st.selectbox('Choose Column where Data is Stored',
                                                              list(trainer['PRED_DATA'].columns))
                        st.success(f'Data Loaded from {trainer["DATA_COLUMN"]}!')

            st.markdown('## Upload Model'
                        'Due to the tendency for model files to be larger than the 200 MB limit of the File Uploader '
                        'Widget, you will need to provide a path to the model. The following text input widget will '
                        'display the current working directory where this app is launched from.')
            trainer['MODEL_PATH'] = st.text_input('Key in the path to the model below',
                                                  value=os.getcwd(),
                                                  key='model_path')
            if os.path.exists(trainer['MODEL_PATH']):
                st.success(f'File Path {trainer["MODEL_PATH"]} exists!')
                trainer['PATH_EXIST'] = True
            else:
                st.error(f'Error: {trainer["MODEL_PATH"]} is invalid!')
                trainer['PATH_EXIST'] = False

            # begin predictions
            st.markdown('## Prediction')
            if st.button('Proceed?'):
                if trainer['PATH_EXIST']:
                    trainer['PRED_DATA'] = trainer['PRED_DATA'][[trainer['DATA_COLUMN']]]
                    trainer['PRED_DATA'] = trainer['PRED_DATA'].to_list()

                    try:
                        trainer['ML_MODEL'] = torch.load('MODEL_PATH')
                        predictions = trainer['ML_MODEL'](trainer['PRED_DATA'])
                    except Exception as ex:
                        st.error(ex)
                    else:
                        st.write(predictions)

                else:
                    st.error('Error: Model File Path is not valid. Try again.')
