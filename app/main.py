from . import config, database, utils


tweets.tweet = tweets.tweet.str.lower()


tweets["clean_tweet"] = tweets.tweet.apply(clean_tweet)

