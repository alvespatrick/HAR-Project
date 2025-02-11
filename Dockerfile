FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir src
COPY src/ src/

RUN mkdir datasets
COPY datasets/ datasets/

RUN mkdir results
RUN mkdir figures

CMD ["python", "src/main.py"]
