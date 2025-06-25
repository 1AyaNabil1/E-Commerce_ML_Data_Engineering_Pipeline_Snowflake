import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from snowflake.snowpark import Session
from snowflake_config import SNOWFLAKE_CONFIG


@st.cache_resource
def create_snowflake_session():
    return Session.builder.configs(SNOWFLAKE_CONFIG).create()


def load_data(session, query):
    return session.sql(query).to_pandas()


def main():
    st.set_page_config(page_title="E-Commerce Analytics Dashboard", layout="wide")

    st.title("🛒 E-Commerce Customer Analytics Dashboard")
    st.markdown("---")

    session = create_snowflake_session()

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    # Total customers
    total_customers = session.sql(
        "SELECT COUNT(*) as count FROM RAW_DATA.RAW_USERS"
    ).collect()[0]["COUNT"]
    col1.metric("Total Customers", f"{total_customers:,}")

    # Total transactions
    total_transactions = session.sql(
        "SELECT COUNT(*) as count FROM RAW_DATA.RAW_TRANSACTIONS"
    ).collect()[0]["COUNT"]
    col2.metric("Total Transactions", f"{total_transactions:,}")

    # Total revenue
    total_revenue = session.sql(
        "SELECT SUM(total_amount) as revenue FROM RAW_DATA.RAW_TRANSACTIONS"
    ).collect()[0]["REVENUE"]
    col3.metric("Total Revenue", f"${total_revenue:,.2f}")

    # Churn rate
    churn_data = load_data(
        session,
        """
        SELECT 
            AVG(CASE WHEN is_churned THEN 1 ELSE 0 END) * 100 as churn_rate
        FROM FEATURES.USER_FEATURES
    """,
    )
    col4.metric("Churn Rate", f"{churn_data['CHURN_RATE'].iloc[0]:.1f}%")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Revenue by Category")
        revenue_by_category = load_data(
            session,
            """
            SELECT p.category, SUM(t.total_amount) as revenue
            FROM RAW_DATA.RAW_TRANSACTIONS t
            JOIN RAW_DATA.RAW_PRODUCTS p ON t.product_id = p.product_id
            GROUP BY p.category
            ORDER BY revenue DESC
        """,
        )

        fig = px.bar(
            revenue_by_category,
            x="CATEGORY",
            y="REVENUE",
            title="Revenue by Product Category",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Customer Segmentation")
        segment_data = load_data(
            session,
            """
            SELECT customer_segment, COUNT(*) as count
            FROM RAW_DATA.RAW_USERS
            GROUP BY customer_segment
        """,
        )

        fig = px.pie(
            segment_data,
            values="COUNT",
            names="CUSTOMER_SEGMENT",
            title="Customer Distribution by Segment",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Churn Analysis
    st.subheader("⚠️ Churn Analysis")

    churn_by_segment = load_data(
        session,
        """
        SELECT 
            u.customer_segment,
            AVG(CASE WHEN f.is_churned THEN 1 ELSE 0 END) * 100 as churn_rate
        FROM RAW_DATA.RAW_USERS u
        JOIN FEATURES.USER_FEATURES f ON u.user_id = f.user_id
        GROUP BY u.customer_segment
        ORDER BY churn_rate DESC
    """,
    )

    fig = px.bar(
        churn_by_segment,
        x="CUSTOMER_SEGMENT",
        y="CHURN_RATE",
        title="Churn Rate by Customer Segment",
    )
    st.plotly_chart(fig, use_container_width=True)

    # High-risk customers
    st.subheader("🚨 High-Risk Customers")
    high_risk = load_data(
        session,
        """
        SELECT u.user_id, u.email, u.customer_segment, f.total_spent
        FROM RAW_DATA.RAW_USERS u
        JOIN FEATURES.USER_FEATURES f ON u.user_id = f.user_id
        WHERE predict_churn(f.age, f.total_transactions, f.total_spent, 
                           f.avg_transaction_amount, f.transaction_frequency) = TRUE
        ORDER BY f.total_spent DESC
        LIMIT 20
    """,
    )

    st.dataframe(high_risk, use_container_width=True)


if __name__ == "__main__":
    main()
