FROM python:3

WORKDIR /usr/src/streamlit
COPY requirements.txt ./
RUN pip install -r ./requirements.txt
RUN python -m nltk.downloader all
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_lg
COPY . .

ENTRYPOINT ["streamlit", "run", "app.py"]