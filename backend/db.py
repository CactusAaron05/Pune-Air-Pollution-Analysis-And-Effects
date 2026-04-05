import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="pune_aqi",
        user="postgres",
        password="mubauman@3"
    )