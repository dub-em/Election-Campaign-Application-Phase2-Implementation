FROM python:3.10.5

WORKDIR /usr/src/app

RUN pip install --upgrade pip

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt

RUN python -m spacy download en

COPY . .

EXPOSE 8000

CMD [ "python", "./app/main.py" ]