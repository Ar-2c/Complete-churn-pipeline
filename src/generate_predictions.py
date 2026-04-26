import pandas as pd
import sqlite3
import pickle
import os

def run_predictions():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, '..', 'Database', 'churn_database.db')
    model_path = os.path.join(base_dir, '..', 'Models', 'final_model.pkl')

    # loading the best model (RF)
    if not os.path.exists(model_path):
        print(f"model not found at {model_path}")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # loading the cleaned data 
    conn = sqlite3.connect(db_path)
    df_cleaned = pd.read_sql("SELECT * FROM prd_churn_cleaned", conn)
    
    # dropping 'Exited'
    X = df_cleaned.drop('Exited', axis=1)

    # generating probabilities and predictions
    # [:, 1] gives us the probability of the positive class (Churn)
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    # attach to raw data
    # We pull the 'stg' data so we have Surnames and Geography names for the UI
    final_display_df = pd.read_sql("SELECT * FROM stg_churn_data", conn)
    final_display_df['Churn_Probability'] = probs
    final_display_df['Predicted_Exited'] = preds

    # saving to the final table (fct_churn_predictions)
    final_display_df.to_sql('fct_churn_predictions', conn, if_exists='replace', index=False)
    conn.close()

    print(f"predictions complete. 'fct_churn_predictions' table created in SQLite.")
    print(f"sample Probabilities: {probs[:5]}")

if __name__ == "__main__":
    run_predictions()