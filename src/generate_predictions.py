import os, urllib.parse, sqlite3, pickle
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
safe_password = urllib.parse.quote_plus(os.getenv("SUPABASE_PASSWORD", ""))
engine = create_engine(f"postgresql+psycopg2://{os.getenv('SUPABASE_USER')}:{safe_password}@{os.getenv('SUPABASE_HOST')}:{os.getenv('SUPABASE_PORT')}/{os.getenv('SUPABASE_DBNAME')}")

# loading the model
model_path = os.path.join(os.path.dirname(__file__), '..', 'Models', 'final_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# loading data for the model (production) and raw data (staging) for the UI
conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), '..', 'Database', 'churn_database.db'))
df_cleaned = pd.read_sql("SELECT * FROM prd_churn_cleaned", conn)
df_stg_clean = pd.read_sql("SELECT * FROM stg_churn_data", conn)
conn.close()

# creating X for the model by droping "exited"
X_clean = df_cleaned.drop('Exited', axis=1, errors='ignore')

# running inference for clean scenario
probs_clean = model.predict_proba(X_clean)[:, 1]
preds_clean = model.predict(X_clean)

# pasting the results in raw data (staging)
df_display_clean = df_stg_clean.copy()
df_display_clean['Churn_Probability'] = probs_clean
df_display_clean['Predicted_Exited'] = preds_clean
df_display_clean['scenario_mode'] = 'CLEAN'

# creating recession scenario, creating the recession in the data for the model first (X)
X_recession = X_clean.copy()
X_recession['Age'] = X_recession['Age'] + 10
X_recession['Balance'] = X_recession['Balance'] * 0.50
X_recession['IsActiveMember'] = 0

# running the model on the "recession" data
probs_recession = model.predict_proba(X_recession)[:, 1]
preds_recession = model.predict(X_recession)

# creating the recession in the raw data as well so it's visible in the UI (data studio)
df_display_recession = df_stg_clean.copy()
df_display_recession['Age'] = df_display_recession['Age'] + 10
df_display_recession['Balance'] = df_display_recession['Balance'] * 0.50
df_display_recession['IsActiveMember'] = 0
df_display_recession['Churn_Probability'] = probs_recession
df_display_recession['Predicted_Exited'] = preds_recession
df_display_recession['scenario_mode'] = 'RECESSION'

# concating the CLEAN and RECESSION so both have staging columns for the UI
df_final = pd.concat([df_display_clean, df_display_recession], ignore_index=True)

# pushing to supabase (if_exists="replace" so old always gets replaced)
df_final.to_sql("fct_churn_predictions", engine, if_exists="replace", index=False)

print("done")