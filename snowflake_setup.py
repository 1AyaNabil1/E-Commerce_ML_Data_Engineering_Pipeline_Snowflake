# snowflake_setup.py
import snowflake.connector
from snowflake_config import SNOWFLAKE_CONFIG

setup_queries = [
    "CREATE DATABASE IF NOT EXISTS ECOMMERCE_DB;",
    "USE DATABASE ECOMMERCE_DB;",
    "CREATE SCHEMA IF NOT EXISTS RAW_DATA;",
    "CREATE SCHEMA IF NOT EXISTS TRANSFORMED;",
    "CREATE SCHEMA IF NOT EXISTS FEATURES;",
    "CREATE SCHEMA IF NOT EXISTS ML_MODELS;",
    """
    CREATE FILE FORMAT IF NOT EXISTS csv_format
    TYPE = 'CSV'
    FIELD_DELIMITER = ','
    RECORD_DELIMITER = '\\n'
    SKIP_HEADER = 1
    FIELD_OPTIONALLY_ENCLOSED_BY = '"';
    """,
    """
    CREATE STAGE IF NOT EXISTS raw_data_stage
    FILE_FORMAT = csv_format;
    """,
]


def run_snowflake_setup():
    print("Connecting to Snowflake...")
    ctx = snowflake.connector.connect(
        user=SNOWFLAKE_CONFIG["user"],
        password=SNOWFLAKE_CONFIG["password"],
        account=SNOWFLAKE_CONFIG["account"],
        warehouse=SNOWFLAKE_CONFIG["warehouse"],
        database=SNOWFLAKE_CONFIG["database"],
        schema=SNOWFLAKE_CONFIG["schema"],
        role=SNOWFLAKE_CONFIG["role"],
    )
    cs = ctx.cursor()

    try:
        for query in setup_queries:
            print(f"Running:\n{query}")
            cs.execute(query)
        print("✅ Setup completed successfully.")
    finally:
        cs.close()
        ctx.close()


if __name__ == "__main__":
    run_snowflake_setup()
