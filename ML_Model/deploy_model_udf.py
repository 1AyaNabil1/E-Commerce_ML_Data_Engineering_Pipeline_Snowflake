import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snowflake.snowpark import Session
from snowflake.snowpark.functions import udf, col
from snowflake.snowpark.types import FloatType, BooleanType
import joblib
import numpy as np
import pandas as pd
from snowflake_config import SNOWFLAKE_CONFIG


def deploy_improved_churn_model():
    print("🚀 Starting model deployment...")
    try:
        session = Session.builder.configs(SNOWFLAKE_CONFIG).create()
        session.file.put(
            "improved_churn_model.pkl", "@ML_MODELS.RAW_DATA_STAGE", auto_compress=False
        )
        model_package = joblib.load("improved_churn_model.pkl")
        model = model_package["model"]
        scaler = model_package["scaler"]
        feature_columns = model_package["feature_columns"]
        model_type = model_package["model_type"]
        print(f"✅ Loaded {model_type} model with features: {feature_columns}")
    except FileNotFoundError:
        print("❌ Model file not found. Please run improved_model_training.py first.")
        return

    try:
        print("🔧 Creating Snowflake UDF...")

        def predict_churn_probability(
            age,
            total_transactions,
            total_spent,
            avg_transaction_amount,
            days_since_last_transaction,
            transactions_last_30_days,
            spend_per_transaction,
            high_value_customer,
            frequent_buyer,
            recency_score,
            payment_method_count,
            customer_segment_encoded,
        ):
            try:
                features = np.array(
                    [
                        [
                            float(age or 0),
                            float(total_transactions or 0),
                            float(total_spent or 0),
                            float(avg_transaction_amount or 0),
                            float(days_since_last_transaction or 0),
                            float(transactions_last_30_days or 0),
                            float(spend_per_transaction or 0),
                            float(high_value_customer or 0),
                            float(frequent_buyer or 0),
                            float(recency_score or 0),
                            float(payment_method_count or 0),
                            float(customer_segment_encoded or 0),
                        ]
                    ]
                )
                if model_type == "LogisticRegression" and scaler:
                    features = scaler.transform(features)
                return float(model.predict_proba(features)[:, 1][0])
            except Exception:
                return 0.5

        session.udf.register(
            func=predict_churn_probability,
            name="predict_churn_probability",
            return_type=FloatType(),
            input_types=[FloatType()] * 12,
            packages=["scikit-learn==1.3.0", "numpy==1.26.4", "pandas==2.0.3"],
            replace=True,
            is_permanent=True,
            stage_location="@ML_MODELS.RAW_DATA_STAGE",
        )
        print("✅ UDF 'predict_churn_probability' registered successfully!")

        def predict_churn_binary(
            age,
            total_transactions,
            total_spent,
            avg_transaction_amount,
            days_since_last_transaction,
            transactions_last_30_days,
            spend_per_transaction,
            high_value_customer,
            frequent_buyer,
            recency_score,
            payment_method_count,
            customer_segment_encoded,
        ):
            try:
                prob = predict_churn_probability(
                    age,
                    total_transactions,
                    total_spent,
                    avg_transaction_amount,
                    days_since_last_transaction,
                    transactions_last_30_days,
                    spend_per_transaction,
                    high_value_customer,
                    frequent_buyer,
                    recency_score,
                    payment_method_count,
                    customer_segment_encoded,
                )
                return prob > 0.5
            except Exception:
                return False

        session.udf.register(
            func=predict_churn_binary,
            name="predict_churn_binary",
            return_type=BooleanType(),
            input_types=[FloatType()] * 12,
            packages=["scikit-learn==1.3.0", "numpy==1.26.4", "pandas==2.0.3"],
            replace=True,
            is_permanent=True,
            stage_location="@ML_MODELS.RAW_DATA_STAGE",
        )
        print("✅ UDF 'predict_churn_binary' registered successfully!")

        print("🧪 Testing UDFs...")
        test_query = f"""
        SELECT 
            user_id,
            age,
            total_transactions,
            total_spent,
            avg_transaction_amount,
            days_since_last_transaction,
            transactions_last_30_days,
            CASE 
                WHEN total_transactions > 0 THEN total_spent / total_transactions 
                ELSE 0 
            END as spend_per_transaction,
            CASE 
                WHEN total_spent > (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY total_spent) FROM FEATURES.USER_FEATURES) 
                THEN 1 ELSE 0 
            END as high_value_customer,
            CASE 
                WHEN total_transactions > (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY total_transactions) FROM FEATURES.USER_FEATURES) 
                THEN 1 ELSE 0 
            END as frequent_buyer,
            recency_score,
            payment_method_count,
            CASE 
                WHEN customer_segment = 'Premium' THEN 2
                WHEN customer_segment = 'Standard' THEN 1
                ELSE 0 
            END as customer_segment_encoded,
            is_churned as actual_churn
        FROM FEATURES.USER_FEATURES 
        LIMIT 10
        """

        features_df = session.sql(test_query).to_pandas()

        prediction_query = f"""
        SELECT 
            user_id,
            age,
            total_transactions,
            total_spent,
            is_churned as actual_churn,
            predict_churn_probability(
                age, 
                total_transactions, 
                total_spent, 
                avg_transaction_amount,
                days_since_last_transaction,
                transactions_last_30_days,
                CASE WHEN total_transactions > 0 THEN total_spent / total_transactions ELSE 0 END,
                CASE WHEN total_spent > (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY total_spent) FROM FEATURES.USER_FEATURES) THEN 1 ELSE 0 END,
                CASE WHEN total_transactions > (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY total_transactions) FROM FEATURES.USER_FEATURES) THEN 1 ELSE 0 END,
                recency_score,
                payment_method_count,
                CASE WHEN customer_segment = 'Premium' THEN 2 WHEN customer_segment = 'Standard' THEN 1 ELSE 0 END
            ) as churn_probability,
            predict_churn_binary(
                age, 
                total_transactions, 
                total_spent, 
                avg_transaction_amount,
                days_since_last_transaction,
                transactions_last_30_days,
                CASE WHEN total_transactions > 0 THEN total_spent / total_transactions ELSE 0 END,
                CASE WHEN total_spent > (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY total_spent) FROM FEATURES.USER_FEATURES) THEN 1 ELSE 0 END,
                CASE WHEN total_transactions > (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY total_transactions) FROM FEATURES.USER_FEATURES) THEN 1 ELSE 0 END,
                recency_score,
                payment_method_count,
                CASE WHEN customer_segment = 'Premium' THEN 2 WHEN customer_segment = 'Standard' THEN 1 ELSE 0 END
            ) as churn_prediction
        FROM FEATURES.USER_FEATURES 
        LIMIT 10
        """

        result = session.sql(prediction_query)
        print("📊 Prediction Results:")
        result.show()

        print("📋 Creating prediction view...")
        view_query = f"""
        CREATE OR REPLACE VIEW ML_MODELS.CUSTOMER_CHURN_PREDICTIONS AS
        SELECT 
            f.*,
            predict_churn_probability(
                f.age, 
                f.total_transactions, 
                f.total_spent, 
                f.avg_transaction_amount,
                f.days_since_last_transaction,
                f.transactions_last_30_days,
                CASE WHEN f.total_transactions > 0 THEN f.total_spent / f.total_transactions ELSE 0 END,
                CASE WHEN f.total_spent > (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY total_spent) FROM FEATURES.USER_FEATURES) THEN 1 ELSE 0 END,
                CASE WHEN f.total_transactions > (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY total_transactions) FROM FEATURES.USER_FEATURES) THEN 1 ELSE 0 END,
                f.recency_score,
                f.payment_method_count,
                CASE WHEN f.customer_segment = 'Premium' THEN 2 WHEN f.customer_segment = 'Standard' THEN 1 ELSE 0 END
            ) as churn_probability,
            predict_churn_binary(
                f.age, 
                f.total_transactions, 
                f.total_spent, 
                f.avg_transaction_amount,
                f.days_since_last_transaction,
                f.transactions_last_30_days,
                CASE WHEN f.total_transactions > 0 THEN f.total_spent / f.total_transactions ELSE 0 END,
                CASE WHEN f.total_spent > (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY total_spent) FROM FEATURES.USER_FEATURES) THEN 1 ELSE 0 END,
                CASE WHEN f.total_transactions > (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY total_transactions) FROM FEATURES.USER_FEATURES) THEN 1 ELSE 0 END,
                f.recency_score,
                f.payment_method_count,
                CASE WHEN f.customer_segment = 'Premium' THEN 2 WHEN f.customer_segment = 'Standard' THEN 1 ELSE 0 END
            ) as churn_prediction
        FROM FEATURES.USER_FEATURES f
        """

        session.sql(view_query).collect()
        print("✅ View created successfully!")

        print("📈 Model Performance Summary:")
        summary_query = """
        SELECT 
            COUNT(*) as total_customers,
            SUM(CASE WHEN churn_prediction THEN 1 ELSE 0 END) as predicted_churned,
            SUM(CASE WHEN is_churned THEN 1 ELSE 0 END) as actual_churned,
            AVG(churn_probability) as avg_churn_probability
        FROM ML_MODELS.CUSTOMER_CHURN_PREDICTIONS
        """

        summary = session.sql(summary_query)
        summary.show()

        print("🎉 Model deployment completed successfully!")
        print("📍 You can now use:")
        print("   - predict_churn_probability() function for probability scores")
        print("   - predict_churn_binary() function for binary predictions")
        print("   - ML_MODELS.CUSTOMER_CHURN_PREDICTIONS view for all predictions")

    except Exception as e:
        print(f"❌ Error during deployment: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        session.close()


if __name__ == "__main__":
    deploy_improved_churn_model()
