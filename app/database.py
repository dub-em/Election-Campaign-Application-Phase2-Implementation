import psycopg2
from psycopg2.extras import RealDictCursor
from .config import settings
import time

def database_connection():
    """A function to connect to the Postgres Remote Instance using the psycopg2 driver."""
    while True:
        try:
            conn = psycopg2.connect(host=settings.database_hostname, database=settings.database_name, 
                                    user=settings.database_user, password=settings.database_password, cursor_factory=RealDictCursor)
            cursor = conn.cursor()
            print('Database connection was successful!')
            break
        except Exception as error:
            print('Connecting to Database failed!')
            print('Error:', error)
            time.sleep(2)
    return (conn, cursor)