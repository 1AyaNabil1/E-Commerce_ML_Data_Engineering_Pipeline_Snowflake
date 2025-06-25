import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
import joblib
from snowflake.snowpark import Session
from snowflake_config import SNOWFLAKE_CONFIG


def train_improved_churn_model():
    print("🚀 Starting improved model training...")

    session = Session.builder.configs(SNOWFLAKE_CONFIG).create()

    try:
        print("📥 Loading improved USER_FEATURES table...")
        features_df = session.table("FEATURES.USER_FEATURES").to_pandas()
        features_df.columns = [col.lower() for col in features_df.columns]

        print(f"✅ Loaded {len(features_df)} user records")
        print(f"📊 Columns: {list(features_df.columns)}")

        print("\n🎯 Churn Distribution:")
        churn_counts = features_df["is_churned"].value_counts()
        print(churn_counts)
        print(
            f"Churn Rate: {(churn_counts.get(True, 0) / len(features_df)) * 100:.2f}%"
        )

        print("🛠️ Engineering additional features...")
        features_df["spend_per_transaction"] = features_df["total_spent"] / features_df[
            "total_transactions"
        ].replace(0, 1)
        features_df["high_value_customer"] = (
            features_df["total_spent"] > features_df["total_spent"].quantile(0.8)
        ).astype(int)
        features_df["frequent_buyer"] = (
            features_df["total_transactions"]
            > features_df["total_transactions"].quantile(0.8)
        ).astype(int)
        features_df["customer_segment_encoded"] = features_df["customer_segment"].map(
            {"Premium": 2, "Standard": 1, "Basic": 0}
        )

        feature_columns = [
            "age",
            "total_transactions",
            "total_spent",
            "avg_transaction_amount",
            "days_since_last_transaction",
            "transactions_last_30_days",
            "spend_per_transaction",
            "high_value_customer",
            "frequent_buyer",
            "recency_score",
            "payment_method_count",
            "customer_segment_encoded",
        ]

        print("🔍 Preparing feature matrix...")
        X = features_df[feature_columns].fillna(0)
        y = features_df["is_churned"].astype(int)

        if y.sum() < 10:
            print(
                "❌ Insufficient churned samples for training. Need at least 10 churned users."
            )
            return None, None

        print("📊 Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("⚖️ Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight="balanced",
            ),
            "LogisticRegression": LogisticRegression(
                random_state=42, class_weight="balanced", max_iter=1000
            ),
        }

        best_model = None
        best_score = 0
        best_model_name = ""

        print("🤖 Training multiple models...")
        for name, model in models.items():
            print(f"\n🔹 Training {name}...")

            if name == "LogisticRegression":
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

            print(f"Accuracy: {accuracy:.3f}")
            print(f"AUC Score: {auc_score:.3f}")
            print(
                f"Cross-Validation AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
            )
            print("Classification Report:")
            print(classification_report(y_test, y_pred))

            if auc_score > best_score:
                best_score = auc_score
                best_model = model
                best_model_name = name

        print(f"\n🏆 Best Model: {best_model_name} (AUC: {best_score:.3f})")

        if best_model_name == "RandomForest":
            print("\n📊 Feature Importance:")
            feature_importance = pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance": best_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            print(feature_importance)

        print("💾 Saving model...")
        model_package = {
            "model": best_model,
            "scaler": scaler if best_model_name == "LogisticRegression" else None,
            "feature_columns": feature_columns,
            "model_type": best_model_name,
        }

        joblib.dump(model_package, "improved_churn_model.pkl")

        print("✅ Model training completed successfully!")
        return model_package, feature_columns

    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return None, None

    finally:
        session.close()


if __name__ == "__main__":
    model_package, features = train_improved_churn_model()
    if model_package:
        print("🎉 Training completed successfully!")
    else:
        print("❌ Training failed!")
