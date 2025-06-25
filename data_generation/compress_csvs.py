import gzip
import shutil
import os

csv_files = ["data/users.csv", "data/products.csv", "data/transactions.csv"]

for file in csv_files:
    if os.path.exists(file):
        output_path = file + ".gz"
        with open(file, "rb") as f_in:
            with gzip.open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Compressed: {output_path}")
    else:
        print(f"Not found: {file}")
