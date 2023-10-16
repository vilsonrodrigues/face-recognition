# pip install psycopg2-binary
import os
import psycopg2


def create_connection() -> psycopg2.connect:
    """Create a Postregres connection"""

    DB_HOST = os.getenv("POSTEGRESQL_DB_HOST")
    DB_NAME = os.getenv("POSTEGRESQL_DB_NAME")
    DB_USER = os.getenv("POSTEGRESQL_DB_USER")
    DB_PASSWORD = os.getenv("POSTEGRESQL_DB_PASSWORD")

    return psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        dbname=DB_NAME,
    )
