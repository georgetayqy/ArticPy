"""
This module is used to discriminate between a piece of fake news and real news
"""

# -------------------------------------------------------------------------------------------------------------------- #
# |                                         IMPORT RELEVANT LIBRARIES                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
import torch.nn as nn
import streamlit as st
import torch
import pathlib
import sys

sys.path.append('../')

from pytorch_pretrained_bert import BertTokenizer, BertModel
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from utils import csp_downloaders
from utils.helper import readFile

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                GLOBAL VARIABLES                                                  | #
# -------------------------------------------------------------------------------------------------------------------- #
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
DOWNLOAD_PATH = (STREAMLIT_STATIC_PATH / "downloads")
DATA = pd.DataFrame
DATA_COL = []
DATA_COLUMN = 'text'
FILE_MODE = 'Small File(s)'
FILE_FORMAT = 'CSV'
DATA_PATH = None
CSP = None
SAVE = False
VERBOSE = False
MODE = None
CONCATED_DATA = pd.DataFrame()
DROPOUT_VALUE = 0.20
LINEAR_IN = 768
LINEAR_OUT = 1
LEARNING_RATE = 0.000001
BATCH_SIZE = 1
EPOCH = 10


# -------------------------------------------------------------------------------------------------------------------- #
# |                                             MAIN APP FUNCTIONALITY                                               | #
# -------------------------------------------------------------------------------------------------------------------- #
def app():
    """
    This function allows the user to train models that will be used for the verification of news article on their
    authenticity and legitimacy
    """

# -------------------------------------------------------------------------------------------------------------------- #
# |                                               GLOBAL VARIABLES                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    global FILE_MODE, FILE_FORMAT, DATA_PATH, DATA, CSP, SAVE, VERBOSE, MODE, CONCATED_DATA, \
        BERTBinaryClassifier, DOWNLOAD_PATH, DROPOUT_VALUE, LINEAR_IN, LINEAR_OUT, LEARNING_RATE, BATCH_SIZE, EPOCH, \
        DATA_COLUMN, DATA_COL

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                    INIT                                                          | #
# -------------------------------------------------------------------------------------------------------------------- #
    st.title('News Discriminator')
    torch.cuda.set_device(0)
    st.markdown('## Init\n'
                'This module is used to discriminate a piece of fake and real news using BERT. For this module, '
                'ensure that your data is decently cleaned (have any wrongly encoded text converted to ASCII '
                'characters.\n\n')

    st.markdown('## Data Selector')
    FILE_MODE = st.selectbox('Choose the size of the file you wish to upload', ('Small File(s)', 'Large File(s)'))
    FILE_FORMAT = st.selectbox('Choose the format of the file you wish to upload', ('CSV', 'XLSX'))
    MODE = st.selectbox('Choose Training or Evaluation Mode', ('Training', 'Evaluation'))

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                 SMALL FILES                                                      | #
# -------------------------------------------------------------------------------------------------------------------- #
    if FILE_MODE == 'Small File(s)':
        st.markdown('## Load Data\n'
                    'Load up the file containing all the data points you want to use for training and evaluation.')
        DATA_PATH = st.file_uploader('Load up one or more CSV/XLSX File containing the cleaned data',
                                     type=[FILE_FORMAT])

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                    DATA LOADER                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
        if DATA_PATH is not None:
            DATA = readFile(DATA_PATH, FILE_FORMAT)
            DATA_COL = DATA.columns.values.tolist()
            DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', options=DATA_COL)
            if not DATA.empty:
                DATA = DATA.astype(str)
                st.success(f'Dataset Loaded!')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                    LARGE FILES                                                   | #
# -----------------------------------------------------------------------------------------------====----------------- #
    elif FILE_MODE == 'Large File(s)':
        st.markdown('## Load Data\n'
                    'In the selection boxes below, select the Cloud Service Provider which you have stored the '
                    'data you wish to analyse.\n'
                    'Unlike Small File Mode, you are not allowed to upload mutliple data files from your CSP. You '
                    'are to prepare a fully concatenated dataset and upload it to the CSP platform.')
        CSP = st.selectbox('CSP', ('Choose a CSP', 'Azure', 'Amazon', 'Google'))

        # FUNCTIONALITY FOR FILE RETRIEVAL
        if CSP == 'Azure':
            azure = csp_downloaders.AzureDownloader()
            if st.button('Continue', key='az'):
                azure.downloadBlob()
                DATA = readFile(azure.AZURE_DOWNLOAD_ABS_PATH, FILE_FORMAT)
                if not DATA.empty:
                    DATA = DATA.astype(str)
                    DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                    st.info('File Read!')

        elif CSP == 'Amazon':
            aws = csp_downloaders.AWSDownloader()
            if st.button('Continue', key='aws'):
                aws.downloadFile()
                DATA = readFile(aws.AWS_FILE_NAME, FILE_FORMAT)
                if not DATA.empty:
                    DATA = DATA.astype(str)
                    DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                    st.info('File Read!')

        elif CSP == 'Google':
            gcs = csp_downloaders.GoogleDownloader()
            if st.button('Continue', key='gcs'):
                gcs.downloadBlob()
                DATA = readFile(gcs.GOOGLE_DESTINATION_FILE_NAME, FILE_FORMAT)
                if not DATA.empty:
                    DATA = DATA.astype(str)
                    DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                    st.info('File Read!')

        elif CSP == 'Google Drive':
            gd = csp_downloaders.GoogleDriveDownloader()
            if st.button('Continue', key='gd'):
                gd.downloadBlob()
                DATA = readFile(gd.GOOGLE_DRIVE_OUTPUT_FILENAME, FILE_FORMAT)
                if not DATA.empty:
                    DATA = DATA.astype(str)
                    DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                    st.info('File Read!')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                    DATA LOADER                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
    if MODE == 'Training':
        st.markdown('# Training Model\n'
                    'This mode will allow you to train your own model by using your own parameters when building the '
                    'BERT model. Note that not all parameters can be altered as we are using a pre-trained model that '
                    'is built by Google.\n')

        st.markdown('## Model Parameters\n')
        with st.form('Parameters'):
            DROPOUT_VALUE = st.number_input('Dropout Layer Value',
                                            min_value=0.,
                                            max_value=1.,
                                            step=0.01,
                                            value=0.20,
                                            format='%.2f', )
            LINEAR_IN = st.number_input('Linear Layer Input Values',
                                        min_value=1,
                                        max_value=1000,
                                        step=1,
                                        value=768)
            LINEAR_OUT = st.number_input('Linear Layer Output Values',
                                         min_value=1,
                                         max_value=1000,
                                         step=1,
                                         value=1)
            LEARNING_RATE = st.number_input('Learning Rate',
                                            min_value=0.0000001,
                                            max_value=0.00001,
                                            step=0.0000001,
                                            value=0.000001,
                                            format='%.7f')
            BATCH_SIZE = st.number_input('Batch Size',
                                         min_value=1,
                                         max_value=10000,
                                         step=1,
                                         value=1)
            EPOCH = st.number_input('Epochs to Train for',
                                    min_value=1,
                                    max_value=10000,
                                    step=1,
                                    value=10)
            param_submit = st.form_submit_button('Confirm Parameters')

            if param_submit:
                st.info('**Parameters Submitted!**\n\n')
                st.info('**Params Inputted:**\n\n'
                        f'Column Selected: {DATA_COLUMN}\n\n'
                        f'Dropout Layer Value: {DROPOUT_VALUE}\n\n'
                        f'Linear Input Layer Neuron Count: {LINEAR_IN}\n\n'
                        f'Linear Output Layer Neuron Count: {LINEAR_OUT}\n\n'
                        f'Learning Rate: {LEARNING_RATE}\n\n'
                        f'Batch Size: {BATCH_SIZE}\n\n'
                        f'Epoch Count: {EPOCH}')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                  TRAIN BERT MODEL                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
        if st.button('Start Training', key='train'):
            # SHUFFLE THE DATA AROUND BY SAMPLING
            CONCATED_DATA = DATA[[DATA_COLUMN]]
            CONCATED_DATA.dropna(inplace=True)
            CONCATED_DATA = CONCATED_DATA.sample(frac=1).reset_index(drop=True)

            # ASSIGN ALL NEWS ARTICLES WITH LABEL 'FAKE'
            CONCATED_DATA['type'] = 'fake'

            # TRAIN/TEST SPLIT: 70/30
            x_train = CONCATED_DATA[:int(0.7 * len(CONCATED_DATA))]
            x_test = CONCATED_DATA[int(0.7 * len(CONCATED_DATA)):]

            # PREPROCESSING OF TRAINING DATA
            x_train_data = [{'text': text, 'type': type_data} for text in list(x_train) for type_data in
                            list(x_train['type'])]
            x_test_data = [{'text': text, 'type': type_data} for text in list(x_test) for type_data in
                           list(x_test['type'])]
            train_text, train_label = list(zip(*map(lambda d: (d['text'], d['type']), x_train_data)))
            test_text, test_label = list(zip(*map(lambda d: (d['text'], d['type']), x_test_data)))

            # TOKENIZE THE DATA
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:511], train_text))
            test_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:511], test_text))

            train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
            test_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))

            train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=512, truncating="post", padding="post",
                                             dtype="int")
            test_tokens_ids = pad_sequences(test_tokens_ids, maxlen=512, truncating="post", padding="post",
                                            dtype="int")

            # CREATE 2ND DIMENSION OF ARRAY
            y_train = np.array(train_label) == 'fake'
            y_test = np.array(test_label) == 'fake'

            # DEFINE MASKS AND TENSORS FOR THE TOKENIZED TRAINING VALUES
            train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
            test_masks = [[float(i > 0) for i in ii] for ii in test_tokens_ids]

            train_masks_tensor = torch.tensor(train_masks)
            test_masks_tensor = torch.tensor(test_masks)

            train_tokens_tensor = torch.tensor(train_tokens_ids)
            y_train_tensor = torch.tensor(y_train.reshape(-1, 1)).float()
            test_tokens_tensor = torch.tensor(test_tokens_ids)
            y_test_tensor = torch.tensor(y_test.reshape(-1, 1)).float()

            # INIT TRAINER
            train_dataset = torch.utils.data.TensorDataset(train_tokens_tensor, train_masks_tensor, y_train_tensor)
            train_sample = torch.utils.data.RandomSampler(train_dataset)
            train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                           sampler=train_sample,
                                                           batch_size=BATCH_SIZE)

            # INIT EVALUATOR
            test_dataset = torch.utils.data.TensorDataset(test_tokens_tensor, test_masks_tensor, y_test_tensor)
            test_sample = torch.utils.data.SequentialSampler(test_dataset)
            test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                          sampler=test_sample,
                                                          batch_size=BATCH_SIZE)

            # CREATE THE CLASSIFIER CLASS
            class BERTBinaryClassifier(nn.Module):
                def __init__(self, dropout=DROPOUT_VALUE):
                    super(BERTBinaryClassifier, self).__init__()

                    self.bert = BertModel.from_pretrained('bert-base-uncased')
                    self.dropout = nn.Dropout(dropout)
                    self.linear = nn.Linear(LINEAR_IN, LINEAR_OUT)
                    self.sigmoid = nn.Sigmoid()

                def forward(self, tokens, masks=None):
                    _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
                    dropout_output = self.dropout(pooled_output)
                    linear_output = self.linear(dropout_output)
                    probability = self.sigmoid(linear_output)
                    return probability

            # INIT MODEL AND DEFINE PARAMETERS
            bert = BERTBinaryClassifier()
            optimizer = torch.optim.Adam(bert.parameters(), lr=LEARNING_RATE)

            # TRAINING
            for epoch_num in range(EPOCH):
                bert.train()
                train_loss = 0
                for step_num, batch_data in enumerate(train_dataloader):
                    token_ids, masks, labels = tuple(t for t in batch_data)
                    probas = bert(token_ids, masks)
                    loss_func = nn.BCELoss()
                    batch_loss = loss_func(probas, labels)
                    train_loss += batch_loss.item()
                    bert.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    print(f'Epoch: {epoch_num + 1}')
                    print(f'\r{step_num}: {len(x_train_data) / BATCH_SIZE} \nloss: {train_loss / (step_num + 1)}')

            # EVALUATE THE MODEL PRODUCED
            bert.eval()
            bert_predicted = []
            all_logits = []

            # THIS EVALUATES THE MODEL USING THE 30% OF DATA THAT WAS RESERVED FOR TESTING
            with torch.no_grad():
                for step_num, batch_data in enumerate(test_dataloader):
                    token_ids, masks, labels = tuple(t for t in batch_data)
                    logits = bert(token_ids, masks)
                    loss_func = nn.BCELoss()
                    loss = loss_func(logits, labels)
                    numpy_logits = logits.cpu().detach().numpy()

                    bert_predicted += list(numpy_logits[:, 0] > 0.5)
                    all_logits += list(numpy_logits[:, 0])

            # DISPLAY EVAL RESULTS
            st.write(classification_report(y_test, bert_predicted))

            # SAVE THE MODEL
            st.markdown('## Save Generated Model')
            st.markdown('Save the model at (downloads/bert_model.pt)[downloads/bert_model.pt]')
            torch.save(bert.state_dict(), str(DOWNLOAD_PATH / 'bert_model.pt'))

    elif MODE == 'Evaluation':
        if FILE_MODE == 'Small File(s)':
            st.markdown('## Load Data\n'
                        'Load up the file containing all the data points you want to use for training and evaluation.')
            DATA_PATH = st.file_uploader('Load up one or more CSV/XLSX File containing the cleaned data',
                                         type=[FILE_FORMAT])

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                    DATA LOADER                                                   | #
# -------------------------------------------------------------------------------------------------------------------- #
            if st.button('Load Data', key='data'):
                with st.spinner('Reading Data...'):
                    DATA = readFile(DATA_PATH, FILE_FORMAT)
                    if not DATA.empty:
                        DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                        st.success(f'Dataset Loaded!')

# -------------------------------------------------------------------------------------------------------------------- #
# |                                                   LARGE FILES                                                    | #
# -------------------------------------------------------------------------------------------------------------------- #
        elif FILE_MODE == 'Large File(s)':
            st.markdown('## Load Data\n'
                        'In the selection boxes below, select the Cloud Service Provider which you have stored the '
                        'data you wish to analyse.\n'
                        'Unlike Small File Mode, you are not allowed to upload mutliple data files from your CSP. You '
                        'are to prepare a fully concatenated dataset and upload it to the CSP platform.')
            CSP = st.selectbox('CSP', ('Choose a CSP', 'Azure', 'Amazon', 'Google'))

            # FUNCTIONALITY FOR FILE RETRIEVAL
            if CSP == 'Azure':
                azure = csp_downloaders.AzureDownloader()
                if st.button('Continue', key='az'):
                    azure.downloadBlob()
                    DATA = readFile(azure.AZURE_DOWNLOAD_ABS_PATH, FILE_FORMAT)
                    if not DATA.empty:
                        DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                        st.info('File Read!')

            elif CSP == 'Amazon':
                aws = csp_downloaders.AWSDownloader()
                if st.button('Continue', key='aws'):
                    aws.downloadFile()
                    DATA = readFile(aws.AWS_FILE_NAME, FILE_FORMAT)
                    if not DATA.empty:
                        DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                        st.info('File Read!')

            elif CSP == 'Google':
                gcs = csp_downloaders.GoogleDownloader()
                if st.button('Continue', key='gcs'):
                    gcs.downloadBlob()
                    DATA = readFile(gcs.GOOGLE_DESTINATION_FILE_NAME, FILE_FORMAT)
                    if not DATA.empty:
                        DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                        st.info('File Read!')

            elif CSP == 'Google Drive':
                gd = csp_downloaders.GoogleDriveDownloader()
                if st.button('Continue', key='gd'):
                    gd.downloadBlob()
                    DATA = readFile(gd.GOOGLE_DRIVE_OUTPUT_FILENAME, FILE_FORMAT)
                    if not DATA.empty:
                        DATA_COLUMN = st.selectbox('Choose Column where Data is Stored', list(DATA.columns))
                        st.info('File Read!')

        st.markdown('## Load Model\n'
                    'Load up a saved model created from Training mode and evaluate novel data.')
        DATA = st.file_uploader('Upload trained model')

        # CREATE THE CLASSIFIER CLASS
        class BERTBinaryClassifier(nn.Module):
            def __init__(self, dropout=DROPOUT_VALUE):
                super(BERTBinaryClassifier, self).__init__()

                self.bert = BertModel.from_pretrained('bert-base-uncased')
                self.dropout = nn.Dropout(dropout)
                self.linear = nn.Linear(LINEAR_IN, LINEAR_OUT)
                self.sigmoid = nn.Sigmoid()

            def forward(self, tokens, masks=None):
                _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
                dropout_output = self.dropout(pooled_output)
                linear_output = self.linear(dropout_output)
                probability = self.sigmoid(linear_output)
                return probability

        if st.button('Evauluate Model', key='eval'):
            bert = BERTBinaryClassifier()
            try:
                bert.load_state_dict(torch.load(DATA))
            except Exception as e:
                st.error(f'Error: {e}')
            else:
                st.success('Model Loaded!')

            # SHUFFLE THE DATA AROUND BY SAMPLING
            DATA = DATA[[DATA_COLUMN]]
            DATA.dropna(inplace=True)
            DATA = DATA.reset_index(drop=True)

            # ASSIGN ALL NEWS ARTICLES WITH LABEL 'FAKE'
            CONCATED_DATA['type'] = 'unknown'

            # TRAIN/TEST SPLIT: 70/30
            x_test = DATA[DATA_COLUMN].to_list()

            # PREPROCESSING OF TRAINING DATA
            x_test_data = [{'text': text, 'type': type_data} for text in list(x_test[DATA_COLUMN]) for type_data in
                           list(x_test['type'])]

            test_text, test_label = list(zip(*map(lambda d: (d['text'], d['type']), x_test_data)))

            # TOKENIZE THE DATA
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

            test_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:511], test_text))

            test_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))

            test_tokens_ids = pad_sequences(test_tokens_ids, maxlen=512, truncating="post", padding="post",
                                            dtype="int")

            # CREATE 2ND DIMENSION OF ARRAY
            y_test = np.array(test_label) == 'none'

            # DEFINE MASKS AND TENSORS FOR THE TOKENIZED TRAINING VALUES
            test_masks = [[float(i > 0) for i in ii] for ii in test_tokens_ids]

            test_masks_tensor = torch.tensor(test_masks)

            test_tokens_tensor = torch.tensor(test_tokens_ids)
            y_test_tensor = torch.tensor(y_test.reshape(-1, 1)).float()

            # INIT EVALUATOR
            test_dataset = torch.utils.data.TensorDataset(test_tokens_tensor, test_masks_tensor, y_test_tensor)
            test_sample = torch.utils.data.SequentialSampler(test_dataset)
            test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                          sampler=test_sample,
                                                          batch_size=BATCH_SIZE)

            # EVALUATE THE MODEL PRODUCED
            bert.eval()
            bert_predicted = []
            all_logits = []

            # THIS EVALUATES THE MODEL USING THE 30% OF DATA THAT WAS RESERVED FOR TESTING
            with torch.no_grad():
                for step_num, batch_data in enumerate(test_dataloader):
                    token_ids, masks, labels = tuple(t for t in batch_data)
                    logits = bert(token_ids, masks)
                    loss_func = nn.BCELoss()
                    loss = loss_func(logits, labels)
                    numpy_logits = logits.cpu().detach().numpy()

                    bert_predicted += list(numpy_logits[:, 0] > 0.5)
                    all_logits += list(numpy_logits[:, 0])

            # DISPLAY EVAL RESULTS
            st.write(classification_report(y_test, bert_predicted))


# RUN APPLICATION
if __name__ == '__main__':
    app()
