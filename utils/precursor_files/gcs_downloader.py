"""GCC"""
# -------------------------------------------------------------------------------------------------------------------- #
# |                                         IMPORT RELEVANT LIBRARIES                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
import os
import shutil

import streamlit as st
from google.cloud import storage

# -------------------------------------------------------------------------------------------------------------------- #
# |                                             GLOBAL VARIABLES                                                     | #
# -------------------------------------------------------------------------------------------------------------------- #
GOOGLE_APPLICATION_CREDENTIALS = 'codeChallenge_GCC_Authentication.json'
GOOGLE_BUCKET_NAME = 'codechallengebucket'
GOOGLE_STORAGE_OBJECT_NAME = '100k_news.csv'
GOOGLE_DESTINATION_FILE_NAME = os.path.join(os.getcwd(), 'download.csv')


# -------------------------------------------------------------------------------------------------------------------- #
# |                                           MAIN DOWNLOADER CLASS                                                  | #
# -------------------------------------------------------------------------------------------------------------------- #
class GoogleDownloader:
    """
    This class manages the downloading of files stored on Google Cloud

    This class only permits you to download files from a blob, one at a time. No threading is enabled on this class.

    Global Variables
    ----------------
    GOOGLE_APPLICATION_CREDENTIALS:                 Represents the path to the JSON file containing the credentials
    GOOGLE_BUCKET_NAME:                             Name of your GCS bucket
    GOOGLE_STORAGE_OBJECT_NAME:                     Name of the file you stored in the GCS bucket
    GOOGLE_DESTINATION_FILE_NAME:                   Path of the downloaded file, defaults to Current Working Directory
    ----------------
    """

    def __init__(self):
        global GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_BUCKET_NAME, GOOGLE_STORAGE_OBJECT_NAME, \
            GOOGLE_DESTINATION_FILE_NAME
        st.title('Google Downloader')
        if st.checkbox('Define Custom GCS Parameters'):
            GOOGLE_BUCKET_NAME = st.text_input('ID of GCS Bucket')
            GOOGLE_STORAGE_OBJECT_NAME = st.text_input('ID of GCS Object')
            GOOGLE_DESTINATION_FILE_NAME = st.text_input('Download Path')
        GOOGLE_APPLICATION_CREDENTIALS = st.file_uploader('Load Service Account Credentials', type=['JSON'])

        if GOOGLE_APPLICATION_CREDENTIALS:
            try:
                st.info('Loading Credentials...')
                GOOGLE_APPLICATION_CREDENTIALS.seek(0)
                with open('../../credentials/google_credentials.json', 'wb') as f:
                    shutil.copyfileobj(GOOGLE_APPLICATION_CREDENTIALS, f)
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.getcwd(),
                                                                            '../../credentials/google_credentials.json')
            except Exception as e:
                st.error(e)
            else:
                st.success('Successfully loaded!')

            try:
                self.GoogleClient = storage.Client()
                self.GoogleBucket = None
                self.GoogleBlob = None

            except Exception as e:
                st.error(e)
                return

    def downloadBlob(self):
        try:
            self.GoogleBucket = self.GoogleClient.bucket(GOOGLE_BUCKET_NAME)
            self.GoogleBlob = self.GoogleBucket.blob(GOOGLE_STORAGE_OBJECT_NAME)
            self.GoogleBlob.download_to_filename(GOOGLE_DESTINATION_FILE_NAME)
        except Exception as e:
            st.error(e)
        else:
            st.success('File Downloaded!')
