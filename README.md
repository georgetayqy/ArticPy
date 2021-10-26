# codeChallenge
This repository serves as a storage location for scripts needed to for the SPF Code Challenge. This README.md file 
serves as a guide on how to set up the environment and to run the scripts. In-depth explanations on how the modules 
work will be provided in a separate README.md file in the `pyfiles` folder.

## Running the App
There are 3 main ways you can access the functionalities of the app, namely replicating/downloading the repository 
onto your local machine and setting up using `requirements.txt`, setting up the environment using Docker, and 
finally through the web application interface hosted on Streamlit.

### Setting up the app environment using `requirements.txt`
#### Init
The first method of running the app is by manually setting up the app environment by pulling the source code from 
this repository and installing the required dependencies yourself on your machine.

Before running the app, you may wish to set up a Virtual Environment to manage all the packages and dependencies that 
will be downloaded. We recommend using a Conda Environment to manage the dependencies as it is the easiest to set up 
and manage. For development, a Conda Environment with Python 3.9.6 installed was used.

To clone the repository, you may run the command `git clone https://github.com/asdfghjkxd/NewsAnalyserApp` if you 
have Git installed on your system. If not, you may wish to visit the site `https://git-scm.com/` to find out more 
on how to download and install Git on your system. If you do not wish to install Git on your system, you may directly 
download the code through the web interface on Github.

#### Running Streamlit
After activating the Virtual Environment of your choice, navigate to the file location where you stored the local copy 
of this repository through your terminal.

Note that there are two different requirements.txt file you may run. Note that due to conflicting dependencies within 
the code that cannot be easily resolved, we have decided to split the Machine Learning portion of the app into a 
separate requirements.txt file. Run the command `pip install -r requirements.txt` on your terminal to 
download and install all dependencies for the main functionality of the app. Run the command 
`pip install -r requirements_gpu.txt` on your terminal to download and install all dependencies for the machine 
learning functionality of the app. If you are using Conda to manage your Python packages, ensure that the Conda 
environment you created is activated in this step. You must install these requirements in different Conda 
environments or you will run into errors.

After installing the dependencies, run the command `streamlit run app.py` in the same folder as above on your terminal 
to initialise the app.

After initialising the app, navigate to the `localhost` address printed out in the terminal in your web browser.
By default, Streamlit is configured to run on `localhost` on port `8501`.

### Setting up the app environment through Docker 🐳
#### Init
The second method of running the app is by using a premade Docker image containing all the source code and dependencies 
preinstalled.

For this method, you need to install Docker on your machine and enable hardware virtualization on your machine. This 
setting may not be turned on by default, and hence you will need to navigate to your BIOS to enable this functionality. 
If you are unsure how to do so or if your hardware supports virtualisation, kindly refer to your motherboard/CPU's 
instruction manual to enable the feature. You may download and install Docker through the link 
`https://docs.docker.com/get-docker/`.

For Windows users, you will need to install and enable Windows Subsystem Linux (WSL) 2 on your system to allow Docker to 
run on your machine. If you do not install WSL 2 on your system or if it is disabled, you will be prompted by Docker to 
install WSL 2. Following the instructions given by Docker to ensure that WSL 2 and Docker is installed properly.

#### Running Streamlit
After installing Docker, you may use your terminal to execute Docker commands on your machine.

Firstly, key in the command `docker pull asdfghjklxd/news:latest` to pull the latest Docker image for the app from 
Docker Hub. Run the command `docker images` to verify if the image has been pulled successfully. If the pull was 
successful, you should be able to see an image by the name of "asdfghjklxd/news" with the tag "latest" on the list 
generated. If you do not see the image on the list or if the list is empty, run the pull command again and allow the 
image to download into your machine.

Next, you may choose to create and run a Docker container on the Command Line Interface (CLI) or on the Docker Desktop 
GUI. If you choose to create and run the Docker container on the CLI, run the command 
`docker run -it -p 5000:8501 --name news asdfghjklxd/news:latest` (the word after the --name tag can be anything you 
want it to be; it represents the name of the container you will be creating) to create and run the Docker Container.
The tags `-it` is for enabling Interactive Mode in the Docker Container and to display an interactive sudo terminal 
on your system's terminal, while `-p` defines the ports to map from your Docker Host (your current system) to the 
Docker Container.

If you choose to use the Docker Desktop GUI, click on 'Images' and identify the image you have downloaded above. Click 
on the 'Run' button and click on 'Optional Settings'. Key in the following details into the relevant fields:

* **Container Name**: A suitable name you wish to give your Docker Container
* **Ports > Local Host**: A suitable and available port on your device; this port will be used to access the app from 
your device
* **Volumes > Host Path**: Path to a folder on your device to store any saved date from the Docker Container to allow 
persistence of data created in the Container (optional, as files are not passed over to the Docker Host through the 
persisted folder mounted onto the Docker Container)
* **Volumes > Container Path**: Path to a folder on the Container, should be in the format /usr/[PATH] (optional, as 
files are not passed over to the Docker Host through the persisted folder mounted onto the Docker Container)

After that, click on 'Run' and navigate to the localhost address `localhost:[YOUR_PORT_ABOVE]` and you should be able 
to see the app UI. If not, navigate back to the Docker Desktop GUI and go through the above steps again to re-create 
the Docker Container.

### Accessing the app through the Web
For this method, no setup is required from you. We have deployed the app on Streamlit, allowing all users to access the 
app without any prior setup.

To access the app, navigate to the website https://share.streamlit.io/asdfghjkxd/newsanalyserapp/main/app.py.

Note that through this method, you may not be able to handle large databases of news articles to process, due to the 
hardware limitations of the instance created online to run the app. If you have a large database of news articles to 
process, it is highly recommended to go with the other two methods mentioned above, as those methods will allow you to
use your own system resources (CPU, RAM, GPU) to run the app and carry out compute-intensive processes.

## File Upload Size Restrictions
We are aware of Streamlit's inherent limitation on the file size which you can upload onto the app. To overcome this, 
you may choose to host your files through any one of the supported Cloud Service Providers and pull the required files 
from there.