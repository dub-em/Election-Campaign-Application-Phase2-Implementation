import schedule, time
import database, utils
import pandas as pd

def app():
    """Function that extract the raw data from the database,
    cleans it, transforms it, predicts the sentiment using the trained model,
    and loads the cleaned and categorized dataset to the database."""

    #Creates connection with the database and extracts raw data
    conn = database.database_connection()          
    sql_query = pd.read_sql_query("""SELECT * FROM election 
                                     WHERE time_created >= (DATE(NOW()) - INTERVAL '1' DAY) 
                                     LIMIT 20""", conn)

    tweets = pd.DataFrame(sql_query, columns = ['time_created','screen_name', 
                                                'name','tweet','loca_tion',
                                                'descrip_tion','verified',
                                                'followers','geo_enabled',
                                                'retweet_count','truncated',
                                                'lang','likes'])
    conn.close()

    #Cleans the dataset (removing emoji, link etc.)
    tweets.tweet = tweets.tweet.str.lower()
    tweets.loca_tion = tweets.loca_tion.str.lower()
    tweets["clean_tweet"] = tweets.tweet.apply(utils.clean_tweet)

    #Transforms the dataset to its tensor format
    dataset = utils.sent_vect(tweets["clean_tweet"])
    ind_var = utils.final_data(dataset)

    #Classifies each tweet by its sentiment using custom pretrained model
    model = utils.model
    prediction = model.predict(ind_var, verbose=False)
    prediction_class = [utils.sent_dict[str(list(row).index(max(list(row))))] for row in prediction]

    #Add this prediction to the cleaned dataset
    tweets['sentiment'] = prediction_class

    #Extracts the trending topic being discussed amongst citizens
    nation_topic = utils.extract_topics(tweets['clean_tweet'])

    #Connect to the database and loads the cleaned and predicted dataset.
    #Loads predicted sentiment to database
    conn, db = database.sqlalchemy_engine()
    tweets.to_sql('citizen_sentiment', con=db, schema='public', if_exists='append',
                   index=False)
    nation_topic.to_sql('national_discourse', con=db, schema='public', if_exists='append',
                         index=False)
    conn.close()

    #Deletes any tweet older than 8 days
    conn = database.database_connection()
    conn.autocommit = True
    cursor = conn.cursor()       
    sql1 = '''DELETE FROM public.citizen_sentiment 
              WHERE time_created <= (DATE(NOW()) - INTERVAL '8' DAY);'''
    cursor.execute(sql1)
    sql2 = '''DELETE FROM public.national_discourse 
              WHERE date <= (DATE(NOW()) - INTERVAL '8' DAY);'''
    cursor.execute(sql2)
    conn.close()

app()

schedule.every(10).minutes.do(app)
#schedule.every(24).hours.do(app)

while True:
    # Checks whether a scheduled task is pending to run or not
    schedule.run_pending()
    time.sleep(5)