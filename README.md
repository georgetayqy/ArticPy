# ArctiPy    [![GitHub release](https://img.shields.io/github/release/asdfghjkxd/ArticPy?include_prereleases=&sort=semver&color=blue)](https://github.com/asdfghjkxd/ArticPy/releases/)
[![asdfghjkxd - ArticPy](https://img.shields.io/static/v1?label=asdfghjkxd&message=ArticPy&color=blue&logo=github)](https://github.com/asdfghjkxd/ArticPy "Go to GitHub repo")
[![stars - ArticPy](https://img.shields.io/github/stars/asdfghjkxd/ArticPy?style=social)](https://github.com/asdfghjkxd/ArticPy)
[![forks - ArticPy](https://img.shields.io/github/forks/asdfghjkxd/ArticPy?style=social)](https://github.com/asdfghjkxd/ArticPy)

An app built to simplify and condense NLP tasks into one simple yet powerful Interface. 

## Setup
### Clone the Repository
To use this app, simple clone the repository onto your local system, navigate into the directory of the cloned 
repository and run the following commands in your favourite Python Virtual Environment!

```shell
pip install -r requirements.txt

streamlit run app.py
```


### Docker
If you do not wish to set up an Environment to run the app, you may choose to run the app using Docker instead! We have
created pre-made Docker images hosted on Github Packages for you to use. To do so, simply install Docker on the target 
system and run the following commands on Terminal or Powershell:

```shell
docker pull ghcr.io/asdfghjkxd/app:main

docker run -it -p 5000:8501 --name news ghcr.io/asdfghjkxd/app:main
```

The created Docker Container can then be accessed through `localhost` on Port `5000`!

If Command Lines are not your thing, you can do the same using the Docker Desktop GUI! Just follow the steps below to 
set up the Container:

- Open up Terminal or Powershell and key in the command `docker pull ghcr.io/asdfghjkxd/app:main` word for word (we 
  promise this is the only Command Line step in the entire process!)
- After the image has been downloaded, open up the Docker Desktop app
- Click on the _Images_ tab on the sidebar and find the image you have pulled in the above step
- Click on the _Run_ button
- Click on the _Optional Settings_ button
- Enter in the variables you want to use in the fields that appear
  - **Container Name**: Enter a suitable name for the Docker Container you are about to create
  - **Ports > Local Host**: Key in a suitable and available port on your device; this port will be used to access the 
    app through your device (corresponds to **[YOUR_PORT_ABOVE]** below)
  - **Volumes > Host Path**: Path to a folder on your device to store any saved date from the Docker Container to allow 
    persistence of data created in the Container (optional, as files are not passed over to the Docker Host through 
    the persisted folder mounted onto the Docker Container)
  - Volumes > Container Path: Path to a folder on the Container, should be in the format /usr/**[PATH]** (optional, as 
    files are not passed over to the Docker Host through the persisted folder mounted onto the Docker Container)
  - Click on _Run_ and navigate to the localhost address localhost:**[YOUR_PORT_ABOVE]** on your web browser

### Web App
If you do not wish to set up your system at all, and you do not mind using the app through the Internet, you may use 
the app on the website https://share.streamlit.io/asdfghjkxd/articpy/main/app.py! You can quickly access the website 
by scanning the QR Code below too!

![WebApp_QRcode](.assets/qr-code.png)

Do note that the performance of the app is restricted to the resources available for the container used to host 
the app online. If you wish to use a GPU, or if your workflow requires greater computing power, kindly use the 
other methods outlined above to run the app.


## Usage
There are 3 main modules in this app, each performing an important step of the way for NLP analysis.

### Load, Clean and Analyse
This module is the first module you will be using to preprocess your data before conducting further analysis on it.

There are 3 things this module can do: Data Cleaning, Data Modification and Data Query.

- Data Cleaning: Cleans and preprocess your raw data for futher analysis
- Data Modification: Allows you to extract the mentions of countries within the documents or allow you to modify your 
    data inplace
- Data Query: Query for certain strings in the document


### Document-Term Matrix
This module will allow you to create Document-Term Matrix and Word Use Frequency Data.


### NLP Toolkit
This module will allow you to conduct advanced NLP analyses on your processed dataset. You are able to perform the 
following tasks on your dataset:

- Topic Modelling: Models your documents using statistical methods
- Topic Classification: Classify your documents using labels you define
- Sentiment Analysis: Analyse the sentiment of your doucments using statistical methods
- Word Cloud Creation: Creates a word cloud visualisation of the entire collection of documents
- Named Entity Recognition: Conducts NER on your documents
- Position of Speech Tagging: COnducts POS on your documents
- Summary: Summarises each document in the collection to your specified parameters
