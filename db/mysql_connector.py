# pip install mysql-connector-python
import os
import mysql.connector


def create_connection() -> mysql.connector.connect:
    """Create a MySQL connection"""

    DB_HOST = os.getenv("MYSQL_DB_HOST")
    DB_NAME = os.getenv("MYSQL_DB_NAME")
    DB_USER = os.getenv("MYSQL_DB_USER")
    DB_PASSWORD = os.getenv("MYSQL_DB_PASSWORD")

    return mysql.connector.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        connection_timeout=30,
        database=DB_NAME,
    )
