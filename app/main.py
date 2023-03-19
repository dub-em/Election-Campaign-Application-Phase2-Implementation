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
        conn = database.database_connection()          
        sql_query = pd.read_sql_query("""SELECT * FROM election LIMIT 10""", conn)

        tweets = pd.DataFrame(sql_query, columns = ['time_reated','screen_name', 
                                                    'name','tweet','loca_tion',
                                                    'descrip_tion','verified',
                                                    'followers','geo_enabled',
                                                    'retweet_count','truncated',
                                                    'lang','likes'])
        conn.close()

        tweets.tweet = tweets.tweet.str.lower()

        tweets["clean_tweet"] = tweets.tweet.apply(utils.clean_tweet)

        dataset = utils.sent_vect5(tweets["clean_tweet"])
        ind_var = utils.final_data(dataset)

        model = utils.model
        prediction = model.predict(ind_var)

        tweets['sentiment'] = prediction

        conn = database.database_connection()
        tweets.to_sql('citizen_sentiment', con=conn, if_exists='append',
                        index=False)

        conn.autocommit = True
        cursor = conn.cursor()       
        sql1 = '''DELETE FROM citizen_sentiment WHERE time_created < current_timestamp - interval '8' day;'''
        cursor.execute(sql1)
        conn.close()

    def flow_caso(schedule=None):
            """
            this function is for the orchestraction/scheduling of this script
            """
            with Flow("primis",schedule=schedule) as flow:
                Extract_Transform = predict_sentiment()
            return flow

    schedule = IntervalSchedule(
        start_date = datetime.datetime.now() + datetime.timedelta(seconds = 2),
        interval = datetime.timedelta(hours=24)
    )
    flow=flow_caso(schedule)

    flow.run()

app()
