FROM python:3

WORKDIR /usr/src/streamlit
COPY requirements.txt ./
RUN pip install -r ./requirements.txt
RUN python -m nltk.downloader all
COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]