import os
import pyodbc

def create_connection() -> pyodbc.connect:
    """ Create a MSSQL connection """

    DB_IP_PORT = os.getenv('MSSQL_DB_IP_PORT')
    DB_NAME = os.getenv('MSSQL_DB_NAME')
    DB_USER = os.getenv('MSSQL_DB_USER')
    DB_PASSWORD = os.getenv('MSSQL_DB_PASSWORD')
    DB_ENCRYPT = os.getenv('MSSQL_DB_ENCRYPT', default='no')

    connection = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=" + DB_IP_PORT + ";"
        "DATABASE=" + DB_NAME + ";"
        "ENCRYPT=" + DB_ENCRYPT + ";"
        "UID=" + DB_USER + ";"
        "PWD=" + DB_PASSWORD
    )

    return pyodbc.connect(connection)