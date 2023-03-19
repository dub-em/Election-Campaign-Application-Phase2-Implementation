def app():
    from . import database, utils
    from datetime import date
    import datetime
    import pandas as pd
    import numpy as np
    from prefect import Flow,task
    from prefect.schedules import IntervalSchedule

    today = datetime.date.today()
    week_ago = datetime.date.today() - datetime.timedelta(days=7)
    yester = datetime.date.today() - datetime.timedelta(days=1)

    @task(max_retries=3, retry_delay=datetime.timedelta(minutes=30))
    def predict_sentiment():
        """Function that extract the raw data from the database,
        cleans it, transforms it, predicts the sentiment using the trained model,
        and loads the cleaned and categorized dataset to the database."""

        #Creates connection with the database
        conn = database.database_connection()          
        sql_query = pd.read_sql_query("""SELECT * FROM election LIMIT 10""", conn)

        tweets = pd.DataFrame(sql_query, columns = ['time_reated','screen_name', 
                                                    'name','tweet','loca_tion',
                                                    'descrip_tion','verified',
                                                    'followers','geo_enabled',
                                                    'retweet_count','truncated',
                                                    'lang','likes'])
        conn.close()

        #Cleans the dataset (removing emoji, link etc.)
        tweets.tweet = tweets.tweet.str.lower()
        tweets["clean_tweet"] = tweets.tweet.apply(utils.clean_tweet)

        #Transforms the dataset to its tensor format
        dataset = utils.sent_vect5(tweets["clean_tweet"])
        ind_var = utils.final_data(dataset)

        #Classifies each tweet by its sentiment using custom pretrained model
        model = utils.model
        prediction = model.predict(ind_var)

        #Add this prediction to the cleaned dataset
        tweets['sentiment'] = prediction

        #Connect to the database and loads the cleaned and predicted dataset.
        conn = database.database_connection()
        tweets.to_sql('citizen_sentiment', con=conn, if_exists='append',
                        index=False)

        #Deletes any tweet older than 8 days
        conn.autocommit = True
        cursor = conn.cursor()       
        sql1 = '''DELETE FROM citizen_sentiment WHERE time_created < current_timestamp - interval '8' day;'''
        cursor.execute(sql1)
        conn.close()

    def flow_caso(schedule=None):
            """
            This function is for the orchestraction/scheduling of this script
            """
            with Flow("primis",schedule=schedule) as flow:
                sentiment = predict_sentiment()
            return flow

    #Define the interval between app intiation cycle
    schedule = IntervalSchedule(
        start_date = datetime.datetime.now() + datetime.timedelta(seconds = 2),
        interval = datetime.timedelta(hours=24)
    )

    #Create the flow object and calls the run method
    flow=flow_caso(schedule)
    flow.run()

app()
