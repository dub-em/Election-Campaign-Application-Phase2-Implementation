FROM python:3.10.5

WORKDIR /usr/src/app

RUN pip install --upgrade pip

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt

RUN python -m spacy download en

COPY . .

EXPOSE 8000

CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000" ]