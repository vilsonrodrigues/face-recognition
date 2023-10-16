# pip install databricks-sql-connector
import os
from databricks import sql


def create_connection() -> sql.connect:
    """Create a Snowflake connection"""

    DB_SERVER_HOSTNAME = os.getenv("DATABRICKS_DB_SERVER_HOSTNAME")
    DB_HTTP_PATH = os.getenv("DATABRICKS_DB_HTTP_PATH")
    DB_ACCESS_TOKEN = os.getenv("DATABRICKS_DB_ACCESS_TOKEN")

    return sql.connect(
        server_hostname=DB_SERVER_HOSTNAME,
        http_path=DB_HTTP_PATH,
        access_token=DB_ACCESS_TOKEN,
    )