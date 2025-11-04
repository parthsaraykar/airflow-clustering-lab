# Enhanced Airflow Clustering Lab

Advanced Apache Airflow pipeline for credit card customer clustering using machine learning.

## Features
- **EDA Task**: Data exploration with statistics, correlations, and outlier detection
- **Feature Engineering**: Purchase/balance ratios and log transformations  
- **Model Comparison**: K-Means, DBSCAN, Agglomerative Clustering
- **Evaluation Metrics**: Silhouette, Davies-Bouldin, Calinski-Harabasz scores
- **Auto Model Selection**: Best model chosen by performance

## Tech Stack
- Apache Airflow 2.7.0
- Docker & Docker Compose
- Python (pandas, scikit-learn, numpy, kneed)
- PostgreSQL

## Setup
```bash
# Start Airflow
docker-compose up -d

# Install packages
docker-compose exec -u airflow airflow-webserver python -m pip install scikit-learn pandas numpy kneed
docker-compose exec -u airflow airflow-scheduler python -m pip install scikit-learn pandas numpy kneed

# Access UI
# http://localhost:8080
# Login: airflow/airflow
```

## Run Pipeline
1. Toggle `Enhanced_Credit_Card_Clustering_Pipeline` ON
2. Click ▶️ Trigger DAG
3. View results in Graph view → Task logs

## Project Structure
```
Lab_1/
├── docker-compose.yaml
├── dags/
│   ├── airflow.py              # DAG definition
│   ├── src/
│   │   └── lab.py              # ML pipeline functions
│   └── data/
│       ├── file.csv            # Training data
│       └── test.csv            # Test data
└── .gitignore
```

## Author
Parth Saraykar - Northeastern University
