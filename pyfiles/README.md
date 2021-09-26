# News Analyser App

## Introduction
This app is used for the analysis of news articles provided in the format of a CSV file.

To get started, first install the required packages needed to run the app by running the command 
`pip install -r requirements.txt` on your Terminal or Command Prompt. Ensure that you have navigated to the right folder
(the same folder as this Markdown document) on your Terminal or Command Prompt. This should automatically install all 
the dependencies required for this app to run.

## Running the App
To run this app natively, on your own machine using your favourite web browser, run the command `streamlit run app.py`. 
As with above, ensure that you have navigated to the same folder as this Markdown document before running the command 
above. After doing so, your web browser should automatically launch the web app which is hosted on your localhost 
server. If it does not, refer to your Terminal or Command Prompt and look for a string that looks like 
<pre> http://localhost:8501/ </pre> 
whereby the last 4 numbers can be any number from 0 to 9, though by default, the port number (the last 4 numbers) 
default to 8501. Click on the link to open up your web browser and to access the web app.

## Use of Streamlit's Web App
If you wish to access the functionalities provided in this app but you do not wish to download the files in this
repository and run it natively on your own device, you may choose to navigate to the webpage
https://share.streamlit.io/asdfghjkxd/newsanalyserapp/main/app.py to use the app. However, because it is 
run on the web, you must manually download the output file and save it on your device with the relevant filename 
and file extensions.

## Explanation of Modules and the Functions in the App
This section will go from the root folder down into the subfolders, and attempt to explain the functions of the files 
that are stored in the folders. Not all files will be explained some groups of files have similar purposes as explained
in the parent folder explanation.

<hr/>

### _processed_data_ Folder
This folder contains all the CSV files with the processed data files. These files can be loaded up to take a look at 
the processed data. However, since the files contained in this folder is largely from an earlier build of this app and 
hence may not reflect the new changes made to the code.

<hr/>

### _pyfiles_ Folder
This folder contains all the files necessary for the app to function. Files in this folder should be protected and 
be free from modifications. Any erroneous modification done to the files may result in errors in the functionality 
of the app. That being said, if you are a developer and would like to change the functionality of the app, feel free
to do so.

#### _pages_ Subfolder
This folder contains essential files required to run the app. The files stored in this folder should be protected and
be free from modifications. Erroneous modification of the files in this folder may result in errors when running
the app. If you wish to change the functionality of the app and is knowledgeable on how to do so, feel free to change
the code to suit your needs.

####  _pages/document_term_matrix.py_ File
This file is used for the preparation and creation of a Document-Term Matrix. Users must first determine the size and 
data format of the raw data they wish to use and choose the appropriate input file method and format. After loading 
the data, a CountVectorizer object will be instantiated. A "bag-of-words" string will also be created; this string 
contains all the words that are used in the text of interest. This string is then converted to a dictionary and then 
fit-transformed using the CountVectorizer object. The resulting fit-transformed data is then converted to an array 
and then fed into a pandas DataFrame, creating the Document-Term Matrix. This file can then be saved into a CSV file 
and then downloaded into the user's system.

#### _pages/load_clean_visualise.py_ File
This file is used for the loading, cleaning and visualising of raw data used for the app. Users must first determine 
size and data format of the raw data they wish to use and choose the appropriate input file method and format. 
Following that, the user can specify flags and parameters to pass into the app. After this, if the user specified to 
clean the data, two separate copies of the raw data will be created, one for the purpose of creating a cleaned copy of 
the raw data, and the second being a modified copy of the cleaned raw data [this step lemmatizes all the words present 
in the text].<br/>

For the cleaning stage, the steps taken are as such:
1. The raw data is piped into a pandas DataFrame
2. The raw data is duplicated into two identical DataFrames, **DF0** (the DataFrame to be cleaned without modification) 
   and **DF1** (the DataFrame that will have all the words lemmatized)
3. Using for loops and panda's internal iterative functions, we iterate through each row of **DF0**
   1. For each row, we used Python's Regular Expressions library _re_ to systematically remove any unwanted characters,
      leaving only alphanumerics and punctuations
   2. Using pandas' _at_ function, we assign the new expression to the current row of the iteration
   3. This process is repeated until all rows have been iterated through
   4. Note that this dataset should only be used as a primary cleaning step to get rid of illegal characters and not 
      for any sort of NLP task, since the data does not contain only lemmatized or stemmed words
4. Similarly, we iterate through **DF1** using the same method as above
   1. For each row, we used Python's Regular Expressions library _re_ to systematically remove any unwanted characters,
      leaving only alphanumerics
   2. We then instantiate a TextBlob object from the _textblob_ library and pass to it the string stored in the row 
   3. The TextBlob object then assign tags to the words and lemmatizes the words in the string based on the tags
   4. Using pandas' _at_ function, we assign the new expression to the current row of the iteration
   5. This process is repeated until all rows have been iterated through
   6. This data is safe for NLP tasks as all the words have already been lemmatized
5. Empty rows in both **DF0** and **DF1** are then removed.
6. If the above steps are successful, users will then be able to download the data into their system as a CSV file.
7. If the user specifies the `verbose` flag, both datasets will be printed out and displayed onto the user's screen, 
   subjected to the maximum size of DataFrame that can be printed out.

#### _pages/toolkit_discriminator.py_ File
This file is used for the discrimination between fake news and legitimate news, and is also able to discriminate 
between machine-written articles and human-written articles. This module uses BERT from Google. For this module, users 
must use a GPU or TPU accelerated system, as the code uses Tensorflow backend to run the machine learning models needed 
to execute the discrimination function. <br/>

For GPUs, users are warned that only NVIDIA GPUs are permitted due to the restrictions of Tensorflow, which is made 
specifically for the CUDA cores found on NVIDIA GPUs. If you are using an AMD GPU 
https://medium.com/analytics-vidhya/install-tensorflow-2-for-amd-gpus-87e8d7aeb812, you must install specific drivers 
to allow Tensorflow to run. Refer to this article for more information on implementing Tensorflow on an AMD GPU. 
Alternatively, you may wish to spin up a Virtual Machine instance on your favourite Cloud Service Provider to run 
the app. <br/>

After enabling GPU or TPU acceleration, the user may wish to specify their own parameters and flags during the 
machine learning process. Once the parameters are loaded up or redefined, the user may then proceed with the 
Inference and Classification task.

Users may run the discriminator in training mode, and create a model for distinguishing between fake and real news, 
though raw, classified data must be provided for the training to be successful.

#### _pages/toolkit_nlp.py_ File
This file contains all the NLP Functionalities used in the app.

**Word Cloud**
This functionality allows the user to input a CSV or XLSX file containing a bag-of-words representation of the text 
they wish to analyse to create a Word Cloud of the text.
<hr/>

**Summary**
This file is used for creating summaries for text inputted to it.

How this module works
1. The user needs to upload a CSV/XLSX file with the correct format. A template is provided for the user to view 
   and download, though the template is limited to just the CSV format.
2. After filling in the details the user wants to process, the user must then decide on the size of the dataset they
   wish to upload and the file format to process the data in.
3. The user will then be presented with a set of flags and parameters they can modify to change the behaviour of the 
   app.
4. Once the user loads up the data, the text will begin the summarising process
   1. Stopwords are removed first
   2. The loaded data is then iterated through
   3. The necessary helper lists are then created for ever pass of the iteration
   4. The words are then converted to all lowercases and then counted
   5. Only words that are used commonly are retained
   6. The words are then appended together into one large string
5. The resulting data is then printed out to the screen, subjected to the overall size of the dataset loaded
<hr/>

**Sentiment Analysis**
This file is used for sentiment analysis of the text users input into the app. Users can specify whether to run VADER 
or TextBlob for their sentiment analysis, though TextBlob will be used by default if the user does not modify the 
settings of the app. <br/>

Note the following differences between the modules used:
* TextBlob is more well-suited for formal pieces of text, such as reports and news articles
* VADER is more well-suited for informal pieces of text, such as text messages, and is able to better process text 
  containing emojis.

How this module works
1. The user needs to upload a CSV/XLSX file with the correct data format. A template is provided for the user to view 
   and download, though the template is limited to just the CSV format.
2. After filling in the details the user wants to process, the user must then decide on the size of the dataset they
   wish to upload and the file format to process the data in.
3. The user will be presented with a set of flags which the user can modify to change the behaviour of the app, and 
   then be prompted to load the data.
4. Once loaded, stopwords will be removed from the text (this functionality has not been implemented so far)
5. If VADER is enabled, VADER will be used to conduct sentiment analysis on the text, otherwise TextBlob will be used 
   by default. Both implementations of VADER and TextBlob are simplistic, i.e. they obtain the word score from a giant 
   corpus and obtain the average word scores for the entire text or sentences.
6. After the analysis is done, users can then view and save the data into a CSV file.
<hr/>

**Topic Modelling**
This file is used to conduct topic modelling on the document users input into the app. Users are to select the Document 
ID (the index of the text in the data) and model the topic of the text; this is a limitation of the app.

How this module works
1. The user uploads a CSV/XLSX file with the correct data format. A template is provided for the user to view and 
   download, though the template is provided in the CSV format only.
2. After filling in the details the user wants to process, the user must then decide on the size of the dataset they
   wish to upload and the file format to process the data in.
3. The user will be presented with a set of flags which the user can modify to change the behaviour of the app, and 
   then be prompted to load the data.
4. Once loaded, the data will be processed into a gensim Dictionary object. An LDA model will also be initialised.
5. The LDA model will then be trained using the gensim Dictionary.
6. After the model is trained, pyLDAvis will be used to visualise the results of the training.
7. Users can then view and save the data into a CSV file.

<hr/>

#### _pages/multipage_ File
This file is used for the implementation of the multiple page functionality of the app. Do not modify this file.<br/>

<hr/>

### _raw_data_ Folder
This folder contains all the raw data files that will be used for the app. You may safely delete the files in this 
folder and replace it with your own folder or delete the folder entirely.

<hr/>

### _utils_ Folder
This folder contains the precursor files and production files that is used for the implementation of Large File 
Downloader in all of the app's module. Do not alter these files unless you wish to modify the credentials to the 
CSP hosting your data, or if you wish to alter the functionality of the app.

<hr/>

### _app.py_ File
This file is the main runner for the app. You may wish to modify this file so that you are able to add new modules to 
the app. <br/>

To add new modules, add the following of code under the section marked by `# DEFINE THE PAGES AND THE APPS THEY CONTAIN`
. Your code should look like this: <br/>
`app.add_page('[NAME OF YOUR MODULE HERE]', name_of_your_module_file.main_function_of_your_module)` <br/>

Don't forget to import your module into the app, in the same manner as the import statements as defined in the section 
`# CUSTOM PAGE IMPORTS`.