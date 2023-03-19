import re, urllib.request
import tensorflow as tf
import gensim
from gensim.models import word2vec, KeyedVectors
from gensim.models.word2vec import Word2Vec
import glob, pprint, spacy
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis, pyLDAvis.gensim_models

# Unicode for emojis
emojis = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)


#Function for cleaning tweet for transformation and classification
def clean_tweet(tweet):
    """
    Function to clean tweet by:
    - Changing '&' sign to and
    - Removing newlines, carriage returns, links, emojis, handles, hashtags and punctuations

    Parameters:
        tweet (string): The tweet
    """      
            
    tweet = re.sub("@[^\s]+", "",tweet) # removes handles
    tweet = re.sub("\n", " ", tweet) # remove newlines
    tweet = re.sub("\r", "", tweet) # remove carriage returns
    tweet = re.sub(r"http\S+", "", tweet) # removes links
    tweet = re.sub(emojis, "", tweet) # remove emojis
    tweet = re.sub(r"#(\w+)", "", tweet) # remove hashtags
    tweet = re.sub("&", "and", tweet) # changes & sign to and
    tweet = re.sub(r"[^\w\s\@]","",tweet) # removes punctuation
    tweet = tweet.strip()

    return tweet

#Function for filtering Tweet
def filter_tweet(tweet, handle, mentions):
    """
    Function that filters tweet Filter for tweets directed at handle, based on the following rules:
    - The handle appears first in tweet.
    - The handle appears in tweet but not after another handle.
    - The person is mentioned any where in tweet based on the list of metions.

    Parameters:
        tweet (string): The tweet
        handle (string): The username of the subject to be filtered for should start with '@'
        mentions (list): A list of other ways the subject could be mentioned in the text
    """

    # Split text into tokens
    tokens = tweet.split()

    # Check for tokens that have the handle
    indices = [i for i, token in enumerate(tokens) if token == handle]

    for index in indices:

        # Checks if the handle appear first
        if index==0:
            return True

        # Checks if the another handle appears before it
        if not tokens[index-1].startswith("@"):
            return True

    # Checks if the person is mentioned anywhere in the tweet
    for mention in mentions:
        if mention in tweet:
            return True
    
    return False


#Loading model from remote repository
try:
    link = './damodel8.h5'
    model = tf.keras.models.load_model(link)
except:
    urllib.request.urlretrieve(
            'https://github.com/dub-em/Election-Campaign-Application-Phase2/raw/main/models/damodel8.h5', 'damodel8.h5')

    link = './damodel8.h5'
    model = tf.keras.models.load_model(link)


wv = KeyedVectors.load('word2vec-google-news-300')