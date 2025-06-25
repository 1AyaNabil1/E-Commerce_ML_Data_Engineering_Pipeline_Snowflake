import schedule
import time
import logging
from datetime import datetime
from data_loader import load_data_to_snowflake
from data_transformation import create_user_features
from model_training import train_churn_model
from deploy_model_udf import deploy_churn_prediction_udf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_daily_pipeline():
    """Run the complete data pipeline"""
    logger.info(f"Starting pipeline run at {datetime.now()}")

    try:
        # Step 1: Load new data (in production, this would be incremental)
        logger.info("Loading data...")
        # load_data_to_snowflake()  # Comment out for demo

        # Step 2: Transform data and create features
        logger.info("Creating features...")
        create_user_features()

        # Step 3: Retrain model (weekly basis)
        if datetime.now().weekday() == 0:  # Monday
            logger.info("Retraining model...")
            train_churn_model()
            deploy_churn_prediction_udf()

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")


def main():
    # Schedule pipeline to run daily at 2 AM
    schedule.every().day.at("02:00").do(run_daily_pipeline)

    logger.info("Pipeline scheduler started...")

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
