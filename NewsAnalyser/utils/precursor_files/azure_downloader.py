"""Azure"""
# -------------------------------------------------------------------------------------------------------------------- #
# |                                         IMPORT RELEVANT LIBRARIES                                                | #
# -------------------------------------------------------------------------------------------------------------------- #
import os
import time
from multiprocessing.pool import ThreadPool

import streamlit as st
from azure.storage.blob import BlobServiceClient

# -------------------------------------------------------------------------------------------------------------------- #
# |                                             GLOBAL VARIABLES                                                     | #
# -------------------------------------------------------------------------------------------------------------------- #
THREADING = False
THREAD_COUNT = 2
AZURE_CONNECTION_STRING = 'DefaultEndpointsProtocol=https;AccountName=codechallengebucket;AccountKey=/rP6m5b8kvQ07tT' \
                          'Ye7NftZIWe0flVw/72a9UDZ6PL+OwNpJH/o20ePijCr04WrXvw4sn4l2putRn3ZPZtuEcZw==;EndpointSuffix=' \
                          'core.windows.net'
AZURE_BLOB_NAME = 'codechallenge'
LOCAL_DOWNLOAD_PATH = os.getcwd()
AZURE_DOWNLOAD_PATH = os.getcwd()


# -------------------------------------------------------------------------------------------------------------------- #
# |                                           MAIN DOWNLOADER CLASS                                                  | #
# -------------------------------------------------------------------------------------------------------------------- #
class AzureDownloader:
    """
    This class manages the downloading of files stored on Azure

    This class also allows you to download just one blob or all blobs stored in your container on Azure Storage

    Global Variables
    ----------------
    THREADING:                      Flag for determining if threading is done
    THREAD_COUNT:                   If THREADING is enabled, this represents the number of threads to use in the
                                    download, hence determining the total number of files to be downloaded at any time
    AZURE_CONNECTION_STRING:        Azure Storage Account Connection String
    AZURE_BLOB_NAME:                Name of Azure Storage Account Blob where data is stored
    LOCAL_DOWNLOAD_PATH:            Local Folder where downloaded data is stored
    AZURE_DOWNLOAD_PATH:            Local Folder path combined with the name of the file, None by default unless User
                                    specifies
    ----------------
    """

    def __init__(self):
        """
        This establishes connection with Azure and defines important ((environment)) variables that is necessary for
        the connection to be open and maintained for the download
        """

        global AZURE_CONNECTION_STRING, AZURE_BLOB_NAME, LOCAL_DOWNLOAD_PATH, THREADING, THREAD_COUNT

        st.title('Azure Downloader')
        if st.checkbox('Advanced Options'):
            THREADING = st.checkbox('Enable Threading?')
            if THREADING:
                THREAD_COUNT = st.number_input('Thread Count',
                                               min_value=1,
                                               max_value=16,
                                               value=2,
                                               step=1)
            AZURE_CONNECTION_STRING = st.text_input("Azure Connection String")
            AZURE_BLOB_NAME = st.text_input("Azure Blob Name")
            LOCAL_DOWNLOAD_PATH = st.text_input("Local Download Path (do not modify if running on web app)",
                                                value=os.path.join(os.getcwd()))

        st.info('Establishing Connection with Azure...')
        time.sleep(2)

        try:
            self.BlobServiceClient_ = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
            self.ClientContainer = self.BlobServiceClient_.get_container_client(AZURE_BLOB_NAME)
            st.success('Connection Established!')
        except Exception as e:
            st.error(f'Error: {e}')
            return

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                            TYPICAL DOWNLOAD                                                  | #
    # ---------------------------------------------------------------------------------------------------------------- #
    def saveBlob(self, filename, file_content):
        """
        Writes the blob content into a file

        Parameters
        ----------
        filename:            Name of the file stored in the blob
        file_content:        Contents of the files stored in the blob
        ----------
        """

        global AZURE_DOWNLOAD_PATH

        # FULL FILEPATH
        AZURE_DOWNLOAD_PATH = os.path.join(LOCAL_DOWNLOAD_PATH, filename)

        # MAKE DIR FOR NESTED BLOBS
        os.makedirs(os.path.dirname(AZURE_DOWNLOAD_PATH), exist_ok=True)

        # WRITE OUT
        with open(AZURE_DOWNLOAD_PATH, 'wb') as azfile:
            azfile.write(file_content)

    def downloadBlob(self):
        """
        Downloads the blob and writes out the name
        """

        ClientBlob = self.ClientContainer.list_blobs()

        for blob in ClientBlob:
            print('.', end='')
            byte = self.ClientContainer.get_blob_client(blob).download_blob().readall()
            self.saveBlob(blob.name, byte)
            st.success('File Successfully downloaded!')

    # ---------------------------------------------------------------------------------------------------------------- #
    # |                                           THREADED DOWNLOAD                                                  | #
    # ---------------------------------------------------------------------------------------------------------------- #
    def threadingSaveBlob(self, blob, file_content):
        """
        Downloads the files contained in the blobs. This function deploys threading.

        Return
        ------
        filename:                   Name of the file stored in the blob
        ------
        """

        global AZURE_DOWNLOAD_PATH

        filename = blob.name
        print(f'Processing {filename}...')
        byte = self.ClientContainer.get_blob_client(blob).download_blob().readall()

        # FULL FILEPATH
        AZURE_DOWNLOAD_PATH = os.path.join(LOCAL_DOWNLOAD_PATH, filename)

        # MAKE DIR FOR NESTED BLOBS
        os.makedirs(os.path.dirname(AZURE_DOWNLOAD_PATH), exist_ok=True)

        # WRITE OUT
        with open(AZURE_DOWNLOAD_PATH, 'wb') as azfile:
            azfile.write(file_content)
        return filename

    def threadingRun(self, blob):
        """
        Instantiates the Thread Pool with the user's specifications. This function deploys threading.
        """

        global THREAD_COUNT

        with ThreadPool(processes=THREAD_COUNT) as threadPool:
            return threadPool.map(self.threadingSaveBlob, blob)

    def threadingDownloadBlob(self):
        """
        Downloads the blob and writes out the name. This function deploys threading.
        """

        ClientBlob = self.ClientContainer.list_blobs()
        ClientResult = self.threadingRun(ClientBlob)
        print(ClientResult)

    if THREADING:
        def downloader(self):
            self.downloadBlob()
    else:
        def downloader(self):
            self.threadingDownloadBlob()