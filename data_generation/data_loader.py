# data_loader.py
import snowflake.connector
from snowflake_setup.snowflake_config import SNOWFLAKE_CONFIG


def load_data_to_snowflake():
    # Connect to Snowflake
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    cursor = conn.cursor()

    try:
        # Upload compressed files to the Snowflake stage in ML_MODELS schema
        cursor.execute(
            "PUT file://data/users.csv.gz @ML_MODELS.RAW_DATA_STAGE AUTO_COMPRESS=FALSE"
        )
        cursor.execute(
            "PUT file://data/products.csv.gz @ML_MODELS.RAW_DATA_STAGE AUTO_COMPRESS=FALSE"
        )
        cursor.execute(
            "PUT file://data/transactions.csv.gz @ML_MODELS.RAW_DATA_STAGE AUTO_COMPRESS=FALSE"
        )

        # Switch to RAW_DATA schema for creating raw tables
        cursor.execute("USE SCHEMA RAW_DATA")

        # Create raw_users table
        cursor.execute("""
            CREATE OR REPLACE TABLE raw_users (
                user_id INTEGER,
                email STRING,
                first_name STRING,
                last_name STRING,
                signup_date DATE,
                country STRING,
                age INTEGER,
                customer_segment STRING
            )
        """)

        # Create raw_products table
        cursor.execute("""
            CREATE OR REPLACE TABLE raw_products (
                product_id INTEGER,
                product_name STRING,
                category STRING,
                price FLOAT,
                brand STRING
            )
        """)

        # Create raw_transactions table
        cursor.execute("""
            CREATE OR REPLACE TABLE raw_transactions (
                transaction_id INTEGER,
                user_id INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                unit_price FLOAT,
                total_amount FLOAT,
                transaction_date TIMESTAMP,
                payment_method STRING
            )
        """)

        # Load data from ML_MODELS.RAW_DATA_STAGE into raw tables
        cursor.execute("""
            COPY INTO raw_users 
            FROM @ML_MODELS.RAW_DATA_STAGE/users.csv.gz 
            FILE_FORMAT = (FORMAT_NAME = ML_MODELS.CSV_FORMAT)
        """)
        cursor.execute("""
            COPY INTO raw_products 
            FROM @ML_MODELS.RAW_DATA_STAGE/products.csv.gz 
            FILE_FORMAT = (FORMAT_NAME = ML_MODELS.CSV_FORMAT)
        """)
        cursor.execute("""
            COPY INTO raw_transactions 
            FROM @ML_MODELS.RAW_DATA_STAGE/transactions.csv.gz 
            FILE_FORMAT = (FORMAT_NAME = ML_MODELS.CSV_FORMAT)
        """)

        print("✅ Data loaded successfully into raw tables!")

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    load_data_to_snowflake()
