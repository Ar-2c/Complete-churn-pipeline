import pandas as pd
import sqlite3
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report

def train_best_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, '..', 'Database', 'churn_database.db')
    model_dir = os.path.join(base_dir, '..', 'Models')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # loading the data
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM prd_churn_cleaned", conn)
    conn.close()

    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # using the RF model with the best parameters from the "modeling_experiments" notebook
    # using class_weight='balanced' based on previous results
    final_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        min_samples_split=5, 
        class_weight='balanced', 
        random_state=42
    )

    print("training final optimized model")
    final_model.fit(X_train, y_train)

    # evaluate for documentation
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"final model results -> F1: {f1:.4f} | AUC: {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # saving the Final Model
    model_path = os.path.join(model_dir, 'final_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)

    print(f"\nmodel saved: {model_path}")

if __name__ == "__main__":
    train_best_model()