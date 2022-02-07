FROM python:3.9.6

WORKDIR /usr/src/streamlit

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["python", "main.py"]
