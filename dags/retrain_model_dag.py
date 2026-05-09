import datetime
import os
import shutil
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from airflow import DAG
from airflow.operators.python import PythonOperator

# Ścieżki
DATA_PATH = "/opt/airflow/data/new_data.csv"
MODELS_DIR = "/opt/airflow/models"
PRODUCTION_DIR = "/opt/airflow/models/production"

def retrain_model(**kwargs):
    # Wczytanie danych
    df = pd.read_csv(DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Podział na treningowy i walidacyjny
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trenowanie modelu
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Walidacja
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Nowy model - Accuracy: {acc:.2f}")

    # Zapis z timestampem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{MODELS_DIR}/rf_model_{timestamp}.pkl"
    joblib.dump(clf, model_path)
    print(f"Model zapisany: {model_path}")

    # Przekazanie danych do następnego taska
    kwargs['ti'].xcom_push(key='model_path', value=model_path)
    kwargs['ti'].xcom_push(key='accuracy', value=acc)

def validate_and_promote(**kwargs):
    ti = kwargs['ti']
    model_path = ti.xcom_pull(key='model_path', task_ids='retrain_model')
    new_acc = ti.xcom_pull(key='accuracy', task_ids='retrain_model')

    production_model = f"{PRODUCTION_DIR}/model.pkl"

    if os.path.exists(production_model):
        old_model = joblib.load(production_model)
        df = pd.read_csv(DATA_PATH)
        X = df.drop("target", axis=1)
        y = df["target"]
        old_acc = accuracy_score(y, old_model.predict(X))
        print(f"Stary model - Accuracy: {old_acc:.2f}")

        if new_acc > old_acc:
            shutil.copy(model_path, production_model)
            print(f"Nowy model wdrożony! Accuracy: {new_acc:.2f} > {old_acc:.2f}")
        else:
            print(f"Stary model lepszy. Zachowano stary. {old_acc:.2f} >= {new_acc:.2f}")
    else:
        # Pierwszy model — od razu wdrażamy
        os.makedirs(PRODUCTION_DIR, exist_ok=True)
        shutil.copy(model_path, production_model)
        print(f"Pierwszy model wdrożony! Accuracy: {new_acc:.2f}")

# Definicja DAG
with DAG(
    dag_id="retrain_model_dag",
    schedule_interval="@daily",
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "retraining"]
) as dag:

    task_retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )

    task_validate = PythonOperator(
        task_id="validate_and_promote",
        python_callable=validate_and_promote,
    )

    task_retrain >> task_validate