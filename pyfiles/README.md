# ArticPy

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
https://share.streamlit.io/asdfghjkxd/articpy/main/app.py to use the app. However, because it is 
run on the web, you must manually download the output file and save it on your device with the relevant filename 
and file extensions.

## Explanation of Modules and the Functions in the App
This section will go from the root folder down into the subfolders, and attempt to explain the functions of the files 
that are stored in the folders. Not all files/folders will be explained in the following section. Only files/folders 
which are important to you are explained below.

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