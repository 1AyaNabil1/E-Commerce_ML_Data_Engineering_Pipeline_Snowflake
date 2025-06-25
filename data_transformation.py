from snowflake.snowpark import Session
from snowflake.snowpark.functions import (
    col,
    count,
    sum as sum_,
    avg,
    max as max_,
    min as min_,
    current_date,
    datediff,
    when,
    lit,
    rand,
)
from snowflake_config import SNOWFLAKE_CONFIG


def create_session():
    return Session.builder.configs(SNOWFLAKE_CONFIG).create()


def create_realistic_user_features(session):
    """Create user-level features with realistic churn detection"""

    print("🔄 Loading base tables...")
    users = session.table("RAW_DATA.RAW_USERS")
    transactions = session.table("RAW_DATA.RAW_TRANSACTIONS")

    print("📊 Computing user transaction features...")
    user_transaction_features = transactions.group_by("user_id").agg(
        [
            count("*").alias("total_transactions"),
            sum_("total_amount").alias("total_spent"),
            avg("total_amount").alias("avg_transaction_amount"),
            max_("total_amount").alias("max_transaction_amount"),
            min_("total_amount").alias("min_transaction_amount"),
            count("transaction_id").alias("transaction_frequency"),
            min_(datediff("day", col("transaction_date"), current_date())).alias(
                "days_since_last_transaction"
            ),
            sum_(
                when(
                    datediff("day", col("transaction_date"), current_date()) <= 30, 1
                ).otherwise(0)
            ).alias("transactions_last_30_days"),
            avg(datediff("day", col("transaction_date"), current_date())).alias(
                "avg_days_between_transactions"
            ),
            # New feature: transaction recency score
            avg(
                1.0 / (datediff("day", col("transaction_date"), current_date()) + 1)
            ).alias("recency_score"),
            # New feature: payment method diversity
            count_distinct("payment_method").alias("payment_method_count"),
        ]
    )

    print("🔗 Joining with user demographics...")
    user_features = users.join(user_transaction_features, "user_id", "inner").select(
        [
            col("user_id"),
            col("age"),
            col("customer_segment"),
            col("total_transactions"),
            col("total_spent"),
            col("avg_transaction_amount"),
            col("max_transaction_amount"),
            col("min_transaction_amount"),
            col("transaction_frequency"),
            col("days_since_last_transaction"),
            col("transactions_last_30_days"),
            col("avg_days_between_transactions"),
            col("recency_score"),
            col("payment_method_count"),
        ]
    )

    print("🎯 Creating realistic churn labels...")
    final_features_with_churn = user_features.with_column(
        "is_churned",
        when(
            (
                (col("days_since_last_transaction") > 90) & (rand() > 0.7)
                | (col("customer_segment") == "Basic") & (rand() > 0.6)
                | (col("age") > 60) & (rand() > 0.8)
            ),
            lit(True),
        ).otherwise(lit(False)),
    )

    print("💾 Saving features to FEATURES.USER_FEATURES...")
    final_features_with_churn.write.save_as_table(
        "FEATURES.USER_FEATURES", mode="overwrite"
    )

    print("📈 Feature Statistics:")
    final_features_with_churn.select(
        [
            count("*").alias("total_users"),
            sum_(when(col("is_churned"), 1).otherwise(0)).alias("churned_users"),
            avg("total_spent").alias("avg_spent"),
            avg("total_transactions").alias("avg_transactions"),
            avg("recency_score").alias("avg_recency_score"),
        ]
    ).show()

    print("🎯 Churn Distribution:")
    churn_dist = final_features_with_churn.group_by("is_churned").agg(
        count("*").alias("count")
    )
    churn_dist.show()

    return final_features_with_churn


def main():
    session = create_session()

    try:
        print("🚀 Starting improved feature creation...")
        user_features = create_realistic_user_features(session)

        print("✅ Features created successfully!")
        print("📋 Sample of features:")
        user_features.show(10)

    finally:
        session.close()


if __name__ == "__main__":
    main()
