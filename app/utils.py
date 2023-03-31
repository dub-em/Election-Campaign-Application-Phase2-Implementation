import pandas as pd
import numpy as np
from datetime import datetime
import re, gensim, spacy
import tensorflow as tf
from gensim.models import KeyedVectors
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#Loading model
#when building the docker image use "./app/models/sentmodel5.h5", because . represents the working
#directory in the Dockerfile, '/usr/src/app' which is the parent folder containing everything
link = './app/models/sentmodel5.h5' 
model = tf.keras.models.load_model(link)


#Loading gensim corpus
#when building the docker image use "./app/corpus/word2vec-google-news-300"
wv = KeyedVectors.load('./app/corpus/word2vec-google-news-300')


#Dictionary to identify the classes in the target variable
sent_dict = {'0':'indifferent','1':'negative','2':'positive'}


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


#Function for transforming the data
def sent_vect(series):
    """This function tokenizes each text and encodes each word in each text with it's vector representation
    in the word2vec-google-news-300 GENSIM dictionary.
    
    This nested list/array will later be converted into a tensor, and fed directly into an RNN"""
    
    shape = series.shape[0]
    series = list(series.values)
    array = []
    pad_array = np.zeros(300)
    for i in range(shape):
        word_token = word_tokenize(series[i])
        sample_vector = np.array([list(wv[word]) for word in word_token if word in wv.index_to_key])
        if sample_vector.shape[0] > 0:
            if sample_vector.shape[0] >= 50:
                sample_vector = sample_vector[:50,:]
            else:
                deficit = 50-sample_vector.shape[0]
                for i in range(deficit):
                    sample_vector = np.vstack((sample_vector, pad_array))
        else:
            sample_vector = np.zeros((50, 300))
        array.append(sample_vector.tolist())
    return array

def final_data(dataset):
    independent_variable = np.array(dataset)
    return independent_variable


#Topic Extraction Functions
def sent_to_word(sentences):
    """ This function removes stop words usign a different method from the precious one(s) used."""
    
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        
def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

#Creates the coprus and dictionary for the topic extraction
def LDA_parameters(text):
    """This function takes the text, processes it, and return the parameters to build the LDA Topic Model."""
    
    data = text.tolist()
    data_words = list(sent_to_word(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    #trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words, bigram_mod)
    data_words_bigrams[:1]

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    return corpus, id2word

#Applies the corpus and dictionary to extract topics, and converts topics to dataframe
def extract_topics(text, num_topics=10):
    corpus, id2word = LDA_parameters(text)

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word,
                                               num_topics=10, random_state=100,
                                               update_every=1, chunksize=100,
                                               passes=10, alpha='auto',
                                               per_word_topics=True)
    
    # arrangement of extracted topics
    topic_words = []
    for i in range(num_topics):
        topic_pair = lda_model.get_topic_terms(i,10)
        topic_words.append([id2word[pair[0]] for pair in topic_pair])

    topic_dict = {'date':[datetime.now() for i in range(10)],
                 'topic importance':['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th'],
                 'word_1':[],'word_2':[],'word_3':[],'word_4':[],'word_5':[],'word_6':[],
                 'word_7':[],'word_8':[],'word_9':[],'word_10':[]}
    
    for topic in topic_words:
        for i in range(1,len(topic)+1):
            string = 'word_' + str(i)
            topic_dict[string].append(topic[i-1])
    topic_df = pd.DataFrame(topic_dict)
    return topic_df