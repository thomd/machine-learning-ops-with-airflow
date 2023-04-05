from datetime import timedelta
from airflow import DAG

from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from tasks import download_dataset, data_processing, ml_training_randomforest, ml_training_logistic, identify_best_model

args = {
    'owner': 'me',
    'retries': 1,
    'start_date': days_ago(1)
}

with DAG(dag_id='ml', default_args=args, schedule=None) as dag:

    task_extract_data = PythonOperator(task_id='download_dataset', python_callable=download_dataset)
    task_process_data = PythonOperator(task_id='data_processing', python_callable=data_processing)
    task_train_rf_model = PythonOperator(task_id='train_randomforest', python_callable=ml_training_randomforest)
    task_train_lr_model = PythonOperator(task_id='train_logistic', python_callable=ml_training_logistic)
    task_identify_best_model = PythonOperator(task_id='identify_best_model', python_callable=identify_best_model)


# workflow process
task_extract_data >> task_process_data >> [task_train_rf_model, task_train_lr_model] >> task_identify_best_model

