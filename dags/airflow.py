# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.lab import (
    load_data, 
    exploratory_data_analysis,
    data_preprocessing, 
    build_compare_models,
    load_model_predict
)

from airflow import configuration as conf

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for your DAG
default_args = {
    'owner': 'parth_mlops',  # Changed owner name
    'start_date': datetime(2025, 11, 1),  # Updated date
    'retries': 1,  # Increased retries
    'retry_delay': timedelta(minutes=3),  # Reduced retry delay
}

# Create a DAG instance with updated name and description
dag = DAG(
    'Enhanced_Credit_Card_Clustering_Pipeline',  # New DAG name
    default_args=default_args,
    description='Advanced clustering pipeline with EDA, feature engineering, and model comparison',
    schedule_interval=None,  # Manual triggering
    catchup=False,
    tags=['clustering', 'mlops', 'credit-card', 'comparison'],  # Added tags for organization
)

# Task 1: Load data from CSV
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

# Task 2: Perform Exploratory Data Analysis (NEW TASK)
eda_task = PythonOperator(
    task_id='exploratory_data_analysis',
    python_callable=exploratory_data_analysis,
    op_args=[load_data_task.output],
    dag=dag,
)

# Task 3: Data preprocessing with feature engineering
preprocessing_task = PythonOperator(
    task_id='feature_engineering_preprocessing',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

# Task 4: Build and compare multiple clustering models
model_comparison_task = PythonOperator(
    task_id='build_compare_clustering_models',
    python_callable=build_compare_models,
    op_args=[preprocessing_task.output, "best_clustering_model.pkl"],
    provide_context=True,
    dag=dag,
)

# Task 5: Load best model and make predictions
prediction_task = PythonOperator(
    task_id='load_model_and_predict',
    python_callable=load_model_predict,
    op_args=["best_clustering_model.pkl", model_comparison_task.output],
    dag=dag,
)

# Set task dependencies
# EDA and preprocessing both depend on load_data, but run independently
# Model comparison depends on preprocessing
# Prediction depends on model comparison
load_data_task >> [eda_task, preprocessing_task]
preprocessing_task >> model_comparison_task >> prediction_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()