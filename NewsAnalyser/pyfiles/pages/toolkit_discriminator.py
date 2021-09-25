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

from pytorch_pretrained_bert import BertTokenizer, BertModel
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from utils import csp_downloaders
from utils.helper import readFile


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


# -------------------------------------------------------------------------------------------------------------------- #
# |                                                      FLAGS                                                       | #
# -------------------------------------------------------------------------------------------------------------------- #
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
DOWNLOAD_PATH = (STREAMLIT_STATIC_PATH / "downloads")
DATA = []


# -------------------------------------------------------------------------------------------------------------------- #
# |                                          MAIN APP FUNCTIONALITY                                                  | #
# -------------------------------------------------------------------------------------------------------------------- #
def app():
    """
    This function allows the user to train models that will be used for the verification of news article on their
    authenticity and legitimacy
    """

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                             GLOBAL VARIABLES                                                 | #
    # ---------------------------------------------------------------------------------------------------------------- #
    global FILE_MODE, FILE_FORMAT, DATA_PATH, DATA, CSP, SAVE, VERBOSE, MODE, CONCATED_DATA, DATA_FIELD, \
        BERTBinaryClassifier, DOWNLOAD_PATH

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                                  INIT                                                        | #
    # ---------------------------------------------------------------------------------------------------------------- #
    st.title('News Discriminator')
    st.markdown('## Init\n'
                'This module is used to discriminate a piece of fake and real news using BERT. For this module, '
                'ensure that your data is decently cleaned (have any wrongly encoded text converted to ASCII '
                'characters.\n\n')

    st.markdown('## Data Selector')
    FILE_MODE = st.selectbox('Choose the size of the file you wish to upload', ('Small File', 'Large File'))
    FILE_FORMAT = st.selectbox('Choose the format of the file you wish to upload', ('CSV', 'XLSX'))

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                              SMALL FILES                                                     | #
    # ---------------------------------------------------------------------------------------------------------------- #
    if FILE_MODE == 'Small File(s)':
        st.markdown('## Load Data\n'
                    'You are allowed to upload more than one file, though the columns and index you use is consistent '
                    'across your data files. You are warned that failure to do so may result in unexpected outcomes '
                    'or errors.')
        DATA_PATH = st.file_uploader('Load up one or more CSV/XLSX File containing the cleaned data',
                                     type=[FILE_FORMAT],
                                     accept_multiple_files=True)

        # ------------------------------------------------------------------------------------------------------------ #
        # |                                                DATA LOADER                                               | #
        # ------------------------------------------------------------------------------------------------------------ #
        if st.button('Load Data', key='data'):
            with st.spinner('Reading Data...'):
                i = 0
                for datadoc in DATA:
                    DATA.append(readFile(datadoc, FILE_FORMAT))
                    if not DATA.empty:
                        i += 1
                        st.success(f'Dataset {i}/{len(DATA)} Loaded!')

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                              LARGE FILES                                                     | #
    # ---------------------------------------------------------------------------------------------------------------- #
    elif FILE_MODE == 'Large File(s)':
        st.markdown('## Load Data\n'
                    'In the selection boxes below, select the Cloud Service Provider which you have stored the data '
                    'you wish to analyse.\n'
                    'Unlike Small File Mode, you are not allowed to upload mutliple data files from your CSP. You '
                    'are to prepare a fully concatenated dataset and upload it to the CSP platform.')
        CSP = st.selectbox('CSP', ('Choose a CSP', 'Azure', 'Amazon', 'Google'))

        # FUNCTIONALITY FOR FILE RETRIEVAL
        if CSP == 'Azure':
            azure = csp_downloaders.AzureDownloader()
            if st.button('Continue', key='az'):
                azure.downloadBlob()
                DATA = readFile(csp_downloaders.AZURE_DOWNLOAD_ABS_PATH, FILE_FORMAT)

        elif CSP == 'Amazon':
            aws = csp_downloaders.AWSDownloader()
            if st.button('Continue', key='aws'):
                aws.downloadFile()
                DATA = readFile(csp_downloaders.AWS_FILE_NAME, FILE_FORMAT)

        elif CSP == 'Google':
            gcs = csp_downloaders.GoogleDownloader()
            if st.button('Continue', key='gcs'):
                gcs.downloadBlob()
                DATA = readFile(csp_downloaders.GOOGLE_DESTINATION_FILE_NAME, FILE_FORMAT)

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                                    FLAGS                                                     | #
    # ---------------------------------------------------------------------------------------------------------------- #
    st.markdown('## Flags')
    SAVE = st.checkbox('Save Output DataFrame into CSV File?')
    VERBOSE = st.slider('Data points to display',
                        min_value=1,
                        max_value=1000,
                        value=20)
    MODE = st.selectbox('Choose data mode', ('Training Mode', 'Novel Data Evaluation Mode'))

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                                 DATA LOADER                                                  | #
    # ---------------------------------------------------------------------------------------------------------------- #
    if MODE == 'Training Mode':
        st.markdown('## Training Mode\n'
                    'This mode will allow you to train your own model by using your own parameters when building '
                    'the BERT model. Note that not all parameters can be altered as we are using a pre-trained '
                    'model that is built by Google.\n')

        st.markdown('## Model Parameters\n')
        param_form = st.form(key='params')
        DROPOUT_VALUE = param_form.number_input('Dropout Layer Value',
                                                min_value=0,
                                                max_value=1,
                                                step=0.01,
                                                value=0.10)
        LINEAR_IN = param_form.number_input('Linear Layer Input Values',
                                            min_value=1,
                                            max_value=1000,
                                            step=1,
                                            value=768)
        LINEAR_OUT = param_form.number_input('Linear Layer Output Values',
                                             min_value=1,
                                             max_value=1000,
                                             step=1,
                                             value=1)
        LEARNING_RATE = param_form.number_input('Learning Rate',
                                                min_value=1e-7,
                                                max_value=1e-5,
                                                step=1e-7,
                                                value=1e-6)
        BATCH_SIZE = param_form.number_input('Batch Size',
                                             min_value=1,
                                             max_value=1e5,
                                             step=1,
                                             value=1)
        EPOCH = param_form.number_input('Epochs to Train for',
                                        min_value=1,
                                        max_value=1e5,
                                        step=1,
                                        value=1)
        param_submit = param_form.form_submit_button('Confirm Parameters')

        # ------------------------------------------------------------------------------------------------------------ #
        # |                                             TRAIN BERT MODEL                                             | #
        # ------------------------------------------------------------------------------------------------------------ #
        if st.button('Start Training', key='train'):
            if param_submit:
                if len(DATA) > 1:
                    # CONCAT DATA TOGETHER UNLESS ONLY ONE FILE WAS PASSED IN
                    try:
                        CONCATED_DATA = pd.concat(DATA,
                                                  axis=0,
                                                  join='outer',
                                                  ignore_index=True,
                                                  )
                    except AssertionError:
                        st.error('Error: Columns between Datasets do not match, data concatenation failed.')
                    else:
                        st.success('Concatenation Process Successful!')
                else:
                    CONCATED_DATA = DATA[0]

                # PREPROCESSING THE DATA
                st.markdown('### Select Relevant Data Fields\n'
                            'Select the correct field of your dataframe which contains the news articles you wish to '
                            'analyse. Failure to do so may result in errors.\n\n'
                            'The final dataframe is loaded below for your reference.')
                st.dataframe(CONCATED_DATA.head(10))
                DATA_FIELD = st.selectbox('Data Field', list(CONCATED_DATA.columns))
                CONCATED_DATA = CONCATED_DATA[[DATA_FIELD]]
                CONCATED_DATA.dropna(inplace=True)
                CONCATED_DATA = CONCATED_DATA.sample(frac=1).reset_index(drop=True)

                # ASSIGN ALL NEWS ARTICLES WITH LABEL 'FAKE'
                CONCATED_DATA['type'] = 'fake'

                # TRAIN/TEST SPLIT: 70/30
                x_train = CONCATED_DATA[:(0.7 * len(CONCATED_DATA))].to_list
                x_test = CONCATED_DATA[(0.7 * len(CONCATED_DATA)):].to_list()

                # PREPROCESSING OF TRAINING DATA
                x_train_data = [{'text': text, 'type': type_data} for text in list(x_train[DATA_FIELD]) for type_data in
                                list(x_train['type'])]
                x_test_data = [{'text': text, 'type': type_data} for text in list(x_test[DATA_FIELD]) for type_data in
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
                y_train = np.array(train_labels) == 'fake'
                y_test = np.array(test_labels) == 'fake'

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

                # INIT MODEL AND DEFINE PARAMETERS
                bert = BertBinaryClassifier()
                optimizer = torch.optim.Adam(bert.parameters(), lr=LEARNING_RATE)

                # TRAINING
                for epoch_num in range(EPOCHS):
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
                        st.info('Epoch: ', epoch_num + 1)
                        st.info('\r' + f'{step_num}/{len(train_data) / BATCH_SIZE} loss: {train_loss / (step_num + 1)}')

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

            else:
                st.error('Warning: You have not defined your model parameters and submitted it. Try again.')
    elif MODE == 'Novel Data Evaluation Mode':
        st.markdown('## Novel Data Evaluation Mode')
