## `TODO: To fix the file paths in this document and update the explanations.`


# codeChallenge
This repository serves as a storage location for scripts needed to for the SPF Code Challenge. This README.md file 
serves as a guide on how to set up the environment and to run the scripts.


## Setting up the Environment
It is recommended to set up either a Python virtual environment (venv), or a Conda virtual environment to install
the necessary packages and dependencies needed for the scripts in this Git repo.


### Python Virtual Environment *(venv)*
To set up a venv to install all dependencies needed for this project, run the following commands on a terminal or
command prompt.


#### Creating the Directory
*For Mac:* </br>

Enter the command <code>sudo bash</code> to activate the bash shell with root privileges. This is necessary to ensure 
that the following commands will be able to run without errors. Do ensure that <code>$PATH</code> is properly updated 
to include 

Next, run <code>python3 -m venv MY_DIRECTORY_NAME</code> on the same Terminal window.<br/>

Ensure that you replace MY_DIRECTORY_NAME with the actual name of the directory which you wish to use to store the 
virtual environment.

*For Windows:*</br>

Unlike MacOS, there is no need to activate any shells. The code works natively on the Command Prompt, given that <code>
PATH</code> is properly updated to include Python in it.

Open up the Command Prompt and key in the code <code>python3 -m venv MY_DIRECTORY_NAME</code> into it and run it.

Ensure that you replace MY_DIRECTORY_NAME with the actual name of the directory which you wish to use to store the 
virtual environment.


#### Activating the Environment
*For Mac:* <br/>
Run the <code>source MY_DIRECTORY_NAME/bin/activate</code> in your 
Terminal. Note that this command only works in the <code>bash</code> shell, which comes preinstalled on MacOS.

*For Windows:*<br/>
Run the command <code>MY_DIRECTORY_NAME\Scripts\activate.bat</code> in the Command Prompt.


#### Installing Dependencies
After activating the Environment, run the command <code>python -m pip install -r requirements.txt</code>. Ensure that 
you are connected to the Internet for this step as pip will download and install the required packages from PyPi.

#### Note on Development System
The scripts in this repo are developed on MacOS, running a Conda environment with Python 3.9.5 installed.
Unexpected behaviours, package dependency clashes and incompatible packages may occur if the target system has 
differing configurations than the one that is outlined above. Do try to follow the exact instructions provided in this
file to ensure that these errors do not occur.

### Conda Environment *(Conda)* [Recommended]
Anaconda comes in two different installers, with differing sizes of install, namely Miniconda and Anaconda. 
Miniconda is suitable for users with low storage space as Miniconda comes packaged with only the necessary programs to 
run conda and Python commands; there is no GUI and no bundled software. Anaconda is suitable for users with more storage
space and would like to use the other software that comes packaged with Anaconda.

This method is recommended as Conda Environments comes with many of the Data Analysis packages that will be needed in 
this project pre-installed. Though, the aforementioned package install script still needs to be run to ensure that
the packages that are installed meets the same package requirements that was used on the Development System to ensure
maximum compatability.

#### Installing Miniconda
Head over to https://docs.conda.io/en/latest/miniconda.html to find the installers for different OSes.

#### Installing Anaconda
Head over to https://docs.anaconda.com/anaconda/install/index.html to find the installer for different OSes.

## Cloning Repository
To clone the repository, first open up your Terminal or Command Prompt and navigate to the location you wish to store 
the files in this repository.

Next, run the command `git clone https://github.com/asdfghjkxd/codeChallenge`. If you are on Windows, ensure that you
have the Git CLI program installed. On Mac, your Terminal already as the Git functionality built-in, hence there is no 
need to install the Git CLI program.

## Installing Dependencies
To ensure that your system has the necessary packages to run the app, you must install all the packages that are stated 
in the *requirements.txt* file.

To do so, navigate to the root folder of this repository. After that, activate your Conda/venv environment and proceed 
to run the command `pip install -r requirements.txt` on your Terminal or Command Prompt. This should install all the 
dependencies needed to run the app.

## Running the App
To run the app natively on your browser, navigate to the subfolder where the separate apps are located in. To run the
News Analyser app, navigate to scripts > NewsAnalyser. To run the Dino Fun Mayhem app, navigate to scripts > 
DinoFunMayhem.

Once inside the folder, run the command `streamlit run app.py`. This command should open up your browser and run the app
. If it does not, look out for your Terminal or Command Prompt output and navigate to the localhost link provided in the
output.

<hr/>

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
https://share.streamlit.io/asdfghjkxd/codechallengenewsanalyser/main/app.py to use the app. However, because it is 
run on the web, you must manually download the output file and save it on your device with the relevant filename 
and file extensions.

## Explanation of Modules and the Functions in the App
This section will go from the root folder down into the subfolders, and attempt to explain the functions of the files 
that are stored in the folders. Not all files will be explained some groups of files have similar purposes as explained
in the parent folder explanation.

<hr/>

### _processed_data_ Folder
This folder contains all the pickled (binary) files containing all the processed data files. These files can be 
loaded up to take a look at the processed data. However, since the files contained in this folder is largely from 
an earlier build of this app and hence may not reflect the new changes made to the code. Files made from 
the latest build of the app will be marked with the tag [NEW].

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
between machine-written articles and human-written articles. The code written in this file is extracted from the 
grover Github Repo at https://github.com/rowanz/grover. For this module, users must use a GPU or TPU accelerated 
system, as the code uses Tensorflow backend to run the machine learning models needed to execute the discrimination 
function. <br/>

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

#### _pages/toolkit_sentiment.py_ File
This file is used for sentiment analysis of the text users input into the app. Users can specify whether to run VADER 
or TextBlob for their sentiment analysis, though TextBlob will be used by default if the user does not modify the 
settings of the app. <br/>

Note the following differences between the modules used:
* TextBlob is more well-suited for formal pieces of text, such as reports and news articles
* VADER is more well-suited for informal pieces of text, such as text messages, and is able to better process text 
  containing emojis.

How this module works
1. The user needs to upload a CSV/XLSX/PKL file with the correct format. A template is provided for the user to view 
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

#### _pages/toolkit_summariser.py_ File
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

#### _pages/toolkit_topic.py_ File [DEPRECATED]
This file contains code used to extract topics from the text inputted into the app. This module is being deprecated 
and replaced with a new module that does topic extraction.

#### _pages/word_use_frequency_analysis.py_ File
This file contains code that is used to analyse the Document-Term Matrix created in pages/document_term_matrix.py 
and visualise the data contained in the file. The user must specify the size of the raw data and the format of the 
data, before uploading the data.

How this module works:
1. The data is loaded
2. User can then specify flags and parameters to pass into the app to modify the behaviour of the app
3. The data is then read into a pandas DataFrame
4. The relevant flags are then applied to the DataFrame before being returned to the user

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
