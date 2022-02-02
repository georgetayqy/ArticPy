FROM python:3.9.6

WORKDIR /usr/src/streamlit
COPY requirements.txt ./
RUN pip install -r ./requirements.txt
RUN python -m nltk.downloader all
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_lg
COPY . .
EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py"]