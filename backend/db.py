import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="your_databse",
        user="your_user",
        password="your_password"
    )
