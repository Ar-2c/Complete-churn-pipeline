import pandas as pd
import sqlite3
import os

# This is an ad-hoc script that creates the .db that's later used in pipeline
# In a real world setting it's assumed this data already comes from a db and not CSV

def run_ingestion():
    # getting the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # going up one level to the project root, then into 'Data'
    csv_path = os.path.join(base_dir, '..', 'Data', 'Customer-Churn-Records.csv')
    
    # going up one level to the project root, then into 'Database'
    db_dir = os.path.join(base_dir, '..', 'Database')
    db_path = os.path.join(db_dir, 'churn_database.db')

    # creating Database folder if it doesn't exist
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    # reads the data
    if not os.path.exists(csv_path):
        print(f"Error: Could not find CSV at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # connects to SQLite
    conn = sqlite3.connect(db_path)
    
    # load to SQL
    # Using replace if need to create it from scratch again
    df.to_sql('stg_churn_data', conn, if_exists='replace', index=False)
    
    # verifying it works
    row_count = pd.read_sql('SELECT COUNT(*) FROM stg_churn_data', conn).iloc[0,0]
    conn.close()
    
    print(f"Database created at: {os.path.abspath(db_path)}")
    print(f"Loaded {row_count} rows into 'stg_churn_data' table.")

if __name__ == "__main__":
    run_ingestion()