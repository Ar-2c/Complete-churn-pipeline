import pandas as pd
import sqlite3
from sklearn.preprocessing import LabelEncoder
import os

# preprocessing data so it can be used for the model

def preprocess_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, '..', 'Database', 'churn_database.db')
    
    # loading from SQL
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM stg_churn_data", conn)
    conn.close()
    
    # dropping unnecessary columns and 'Complain' as previously explained in exploration.ipynb.
    df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Complain'])
    
    # handling of categorical Data. Gender: Binary (0/1)
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    
    # geography and card type: One-Hot Encoding to create columns like 'Geography_Germany', etc.
    df = pd.get_dummies(df, columns=['Geography', 'Card Type'], drop_first=True)
    
    # saving the processed data back to SQLite in a new table
    # putting replace here in case it need to be run again at some point to reset
    conn = sqlite3.connect(db_path)
    df.to_sql('prd_churn_cleaned', conn, if_exists='replace', index=False)
    conn.close()
    
    print("'prd_churn_cleaned' table saved.")

if __name__ == "__main__":
    preprocess_data()