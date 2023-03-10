from . import database, utils
import pandas as pd


conn = database.database_connection()          
sql_query = pd.read_sql_query("""SELECT * FROM election""", conn)

tweets = pd.DataFrame(sql_query, columns = ['time_reated','screen_name', 
                                            'name','tweet','loca_tion',
                                            'descrip_tion','verified',
                                            'followers','geo_enabled',
                                            'retweet_count','truncated',
                                            'lang','likes'])
conn.close()


tweets.tweet = tweets.tweet.str.lower()

tweets["clean_tweet"] = tweets.tweet.apply(utils.clean_tweet)

model = utils.model

