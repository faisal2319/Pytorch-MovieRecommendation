FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt


COPY mlflow_server/ .
COPY database/ ./database/


RUN mkdir -p /mlflow/artifacts && chmod -R 777 /mlflow
EXPOSE 5000