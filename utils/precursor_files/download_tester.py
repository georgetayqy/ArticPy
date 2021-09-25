import streamlit as st

from aws_downloader import AWSDownloader
from azure_downloader import AzureDownloader
from gcs_downloader import GoogleDownloader


def aws():
    amazon = AWSDownloader()
    if st.button('Proceed', key='Loader'):
        amazon.downloadFile()


def az():
    azure = AzureDownloader()

    if st.button('Proceed', key='Loader'):
        azure.downloadBlob()


def gcs():
    google = GoogleDownloader()
    if st.button('Proceed', key='Loader'):
        google.downloadBlob()


if __name__ == '__main__':
    az()
