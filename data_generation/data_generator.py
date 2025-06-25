# data_generator.py
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

fake = Faker()
np.random.seed(42)


# Generate Users
def generate_users(n=10000):
    users = []
    for i in range(n):
        users.append(
            {
                "user_id": i + 1,
                "email": fake.email(),
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "signup_date": fake.date_between(start_date="-2y", end_date="today"),
                "country": fake.country(),
                "age": random.randint(18, 70),
                "customer_segment": random.choice(["Premium", "Standard", "Basic"]),
            }
        )
    return pd.DataFrame(users)


# Generate Products
def generate_products(n=1000):
    categories = ["Electronics", "Clothing", "Books", "Home", "Sports"]
    products = []
    for i in range(n):
        products.append(
            {
                "product_id": i + 1,
                "product_name": fake.sentence(nb_words=3)[:-1],
                "category": random.choice(categories),
                "price": round(random.uniform(10, 500), 2),
                "brand": fake.company(),
            }
        )
    return pd.DataFrame(products)


# Generate Transactions
def generate_transactions(users_df, products_df, n=100000):
    transactions = []
    for i in range(n):
        user_id = random.choice(users_df["user_id"].tolist())
        product_id = random.choice(products_df["product_id"].tolist())
        price = products_df[products_df["product_id"] == product_id]["price"].iloc[0]

        transactions.append(
            {
                "transaction_id": i + 1,
                "user_id": user_id,
                "product_id": product_id,
                "quantity": random.randint(1, 5),
                "unit_price": price,
                "total_amount": price * random.randint(1, 5),
                "transaction_date": fake.date_time_between(
                    start_date="-1y", end_date="now"
                ),
                "payment_method": random.choice(
                    ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"]
                ),
            }
        )
    return pd.DataFrame(transactions)


# Generate all datasets
if __name__ == "__main__":
    print("Generating sample data...")

    users_df = generate_users(10000)
    products_df = generate_products(1000)
    transactions_df = generate_transactions(users_df, products_df, 100000)

    # Save to CSV
    users_df.to_csv("users.csv", index=False)
    products_df.to_csv("products.csv", index=False)
    transactions_df.to_csv("transactions.csv", index=False)

    print("Data generation complete!")
    print(f"Users: {len(users_df)} records")
    print(f"Products: {len(products_df)} records")
    print(f"Transactions: {len(transactions_df)} records")
