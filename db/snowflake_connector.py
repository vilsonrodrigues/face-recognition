# pip install snowflake-connector-python
import os
import snowflake.connector

def create_connection() -> snowflake.connector:
    """Create a Snowflake connection"""

    DB_ACCOUNT = os.getenv("SNOWFLAKE_DB_ACCOUNT")
    DB_NAME = os.getenv("SNOWFLAKE_DB_NAME")
    DB_USER = os.getenv("SNOWFLAKE_DB_USER")
    DB_PASSWORD = os.getenv("SNOWFLAKE_DB_PASSWORD")

    return snowflake.connector.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        account=DB_ACCOUNT,
        database=DB_NAME,
    )
