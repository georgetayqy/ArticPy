#codeChallenge
This repository serves as a storage location for scripts needed to for the SPF Code Challenge. This README.md file 
serves as a guide on how to set up the environment and to run the scripts.

## Running the App
There are many ways you can access the functionalities of the app.

### Setting up the app environment yourself through Github
#### Init
The first method of running the app is by manually setting up the app environment by pulling the app source code and 
installing the required dependencies yourself.

Before running the app, you may wish to set up a Virtual Environment to manage all the packages and dependencies that 
will be downloaded. We recommend using a Conda Environment to manage the dependencies as it is the easiest to set up 
and manage. For development, a Conda Environment with Python 3.9.6 installed was used.

#### Running Streamlit
After activating the Virtual Environment of your choice, navigate to the file location where you stored the local copy 
of this repository. Run the command `pip install -r requirements.txt` on your terminal to download and install all 
dependencies.

After installing the dependencies, run the command `streamlit run app.py` on your terminal to initialise the app.

After initialising the app, navigate to the localhost address printed out in the terminal in your web browser.
By default, streamlit is configured to run on `localhost` on port `8501`.

### Setting up the app environment through Docker
#### Init
The second method of running the app is by using a premade Docker image containing all the source code and dependencies 
preinstalled.

For this method, you need to install Docker on your machine and enable hardware virtualization on your machine. This 
setting may not be turned on by default, and hence you will need to navigate to your BIOS to enable this functionality. 

For Windows users, you will need to install and enable Windows Subshell Linux (WSL) 2 on your system to allow Docker to 
run on your machine.

#### Running Streamlit
After installing Docker, you may use your terminal to execute Docker commands on your machine.

Firstly, key in the command `docker pull asdfghjklxd/news:latest` to pull the latest Docker image for the app from 
Docker Hub. Run the command `docker images` to verify if the image has been pulled successfully. If the pull was 
successful, you should be able to see an image by the name of "asdfghjklxd/news" with the tag "latest" on the list 
generated. If you do not see the image on the list or if the list is empty, run the pull command again and allow the 
image to download into your machine.

Next, you may choose to create and run a Docker container on the Command Line Interface or on the Docker Desktop GUI. 
If you choose to create and run the Docker container on the CLI, run the command `docker run -it -p 5000:8501 --name 
news asdfghjklxd/news:latest` (the word after the --name tag can be anything you want it to be; it represents the 
name of the container you will be creating) to create and run the Docker Container. 

If you choose to use the Docker Desktop GUI, click on 'Images' and identify the image you have downloaded above. Click 
on the 'Run' button and click on 'Optional Settings'. Key in the following details into the relevant fields:

* **Container Name**: A suitable name you wish to give your Docker Container
* **Ports > Local Host**: A suitable and available port on your device; this port will be used to access the app from 
your device
* **Volumes > Host Path**: Path to a folder on your device to store any saved date from the Docker Container to allow 
persistence of data created in the Container
* **Volumes > Container Path**: Path to a folder on the Container, should be in the format /usr/[PATH]

After that, click on 'Run' and navigate to the localhost address `localhost:[YOUR_PORT_ABOVE]` and you should be able 
to see the app UI. If not, navigate back to the Docker Desktop GUI and go through the above steps again to re-create 
the Docker Container.

### Accessing the app through the Web
For this method, no setup is required from you. We have deployed the app on Streamlit, allowing all users to access the 
app without any prior setup.

To access the app, navigate to the website https://share.streamlit.io/asdfghjkxd/newsanalyserapp/main/app.py.