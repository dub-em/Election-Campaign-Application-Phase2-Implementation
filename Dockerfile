FROM python:3.9.16

WORKDIR /usr/src/app

RUN pip install --upgrade pip

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt

COPY . .

CMD [ "python", "./app/main.py" ]