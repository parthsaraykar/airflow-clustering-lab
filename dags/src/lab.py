import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kneed import KneeLocator
import pickle
import os
import json


def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.

    Returns:
        bytes: Serialized data.
    """
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)
    
    return serialized_data


def exploratory_data_analysis(data):
    """
    Performs EDA on the dataset and returns analysis results.

    Args:
        data (bytes): Serialized data to be analyzed.

    Returns:
        dict: Dictionary containing EDA results (statistics, correlations, outliers).
    """
    df = pickle.loads(data)
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Basic statistics
    stats = numeric_df.describe().to_dict()
    
    # Correlation matrix (only numeric columns)
    correlation = numeric_df.corr().to_dict()
    
    # Detect outliers using IQR method (only numeric columns)
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum().to_dict()
    
    # Missing values (all columns)
    missing = df.isnull().sum().to_dict()
    
    # Data shape
    shape = {"rows": df.shape[0], "columns": df.shape[1]}
    
    eda_results = {
        "statistics": stats,
        "correlation": correlation,
        "outliers_count": outliers,
        "missing_values": missing,
        "shape": shape
    }
    
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS RESULTS")
    print("="*50)
    print(f"\nDataset Shape: {shape['rows']} rows x {shape['columns']} columns")
    print(f"\nMissing Values: {missing}")
    print(f"\nOutliers Detected (IQR method): {outliers}")
    print(f"\nCorrelation Matrix:\n{pd.DataFrame(correlation)}")
    print("="*50 + "\n")
    
    return json.dumps(eda_results)

def data_preprocessing(data):
    """
    Deserializes data, performs enhanced preprocessing with feature engineering,
    and returns serialized preprocessed data.

    Args:
        data (bytes): Serialized data to be deserialized and processed.

    Returns:
        bytes: Serialized preprocessed data.
    """
    df = pickle.loads(data)
    
    # Remove missing values
    df = df.dropna()
    
    # Select original features
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]].copy()
    
    # Feature Engineering: Create new meaningful features
    clustering_data['PURCHASE_TO_LIMIT_RATIO'] = clustering_data['PURCHASES'] / (clustering_data['CREDIT_LIMIT'] + 1)
    clustering_data['BALANCE_TO_LIMIT_RATIO'] = clustering_data['BALANCE'] / (clustering_data['CREDIT_LIMIT'] + 1)
    
    # Log transformation for skewed features (add 1 to avoid log(0))
    clustering_data['LOG_BALANCE'] = np.log1p(clustering_data['BALANCE'])
    clustering_data['LOG_PURCHASES'] = np.log1p(clustering_data['PURCHASES'])
    
    print("\n" + "="*50)
    print("DATA PREPROCESSING COMPLETED")
    print("="*50)
    print(f"Original features: BALANCE, PURCHASES, CREDIT_LIMIT")
    print(f"Engineered features: PURCHASE_TO_LIMIT_RATIO, BALANCE_TO_LIMIT_RATIO, LOG_BALANCE, LOG_PURCHASES")
    print(f"Total features for clustering: {clustering_data.shape[1]}")
    print("="*50 + "\n")
    
    # Use RobustScaler instead of MinMaxScaler (more robust to outliers)
    robust_scaler = RobustScaler()
    clustering_data_scaled = robust_scaler.fit_transform(clustering_data)
    
    clustering_serialized_data = pickle.dumps(clustering_data_scaled)
    return clustering_serialized_data


def build_compare_models(data, filename):
    """
    Builds and compares multiple clustering models, saves the best one,
    and returns evaluation metrics.

    Args:
        data (bytes): Serialized data for clustering.
        filename (str): Name of the file to save the best clustering model.

    Returns:
        dict: Dictionary containing SSE values and evaluation metrics for different models.
    """
    df = pickle.loads(data)
    
    # K-Means with elbow method
    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    silhouette_scores = []
    
    print("\n" + "="*50)
    print("EVALUATING K-MEANS FOR DIFFERENT K VALUES")
    print("="*50)
    
    for k in range(2, 20):  # Start from 2 for silhouette score
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
        
        # Calculate silhouette score
        labels = kmeans.labels_
        sil_score = silhouette_score(df, labels)
        silhouette_scores.append(sil_score)
        
        print(f"K={k}: SSE={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
    
    # Find optimal K using elbow method
    kl = KneeLocator(range(2, 20), sse, curve="convex", direction="decreasing")
    optimal_k = kl.elbow if kl.elbow else 5  # Default to 5 if elbow not found
    
    print(f"\nOptimal K from Elbow Method: {optimal_k}")
    print("="*50 + "\n")
    
    # Train final models with optimal K
    print("="*50)
    print(f"COMPARING CLUSTERING ALGORITHMS (K={optimal_k})")
    print("="*50)
    
    models = {
        'kmeans': KMeans(n_clusters=optimal_k, **kmeans_kwargs),
        'agglomerative': AgglomerativeClustering(n_clusters=optimal_k),
        'dbscan': DBSCAN(eps=0.5, min_samples=5)
    }
    
    results = {}
    best_model = None
    best_score = -1
    
    for model_name, model in models.items():
        model.fit(df)
        labels = model.labels_
        
        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        sil_score = silhouette_score(df, labels) if n_clusters > 1 else -1
        db_score = davies_bouldin_score(df, labels) if n_clusters > 1 else -1
        ch_score = calinski_harabasz_score(df, labels) if n_clusters > 1 else -1
        
        results[model_name] = {
            'n_clusters': n_clusters,
            'silhouette_score': sil_score,
            'davies_bouldin_score': db_score,
            'calinski_harabasz_score': ch_score
        }
        
        print(f"\n{model_name.upper()}:")
        print(f"  Clusters: {n_clusters}")
        print(f"  Silhouette Score: {sil_score:.3f}")
        print(f"  Davies-Bouldin Score: {db_score:.3f}")
        print(f"  Calinski-Harabasz Score: {ch_score:.3f}")
        
        # Select best model based on silhouette score
        if sil_score > best_score:
            best_score = sil_score
            best_model = (model_name, model)
    
    print(f"\n{'='*50}")
    print(f"BEST MODEL: {best_model[0].upper()} (Silhouette Score: {best_score:.3f})")
    print("="*50 + "\n")
    
    # Save the best model
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'wb') as f:
        pickle.dump(best_model[1], f)
    
    # Return both SSE list and results dictionary
    return_data = {
        'sse': sse,
        'silhouette_scores': silhouette_scores,
        'optimal_k': optimal_k,
        'model_comparison': results,
        'best_model': best_model[0]
    }
    
    return pickle.dumps(return_data)


def load_model_predict(filename, model_results):
    """
    Loads saved model and makes predictions on test data.

    Args:
        filename (str): Name of the file containing the saved clustering model.
        model_results (bytes): Serialized model evaluation results.

    Returns:
        dict: Prediction results and model information.
    """
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    
    # Load the saved model
    loaded_model = pickle.load(open(output_path, 'rb'))
    
    # Load test data
    test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    
    # Apply same preprocessing to test data
    test_df['PURCHASE_TO_LIMIT_RATIO'] = test_df['PURCHASES'] / (test_df['CREDIT_LIMIT'] + 1)
    test_df['BALANCE_TO_LIMIT_RATIO'] = test_df['BALANCE'] / (test_df['CREDIT_LIMIT'] + 1)
    test_df['LOG_BALANCE'] = np.log1p(test_df['BALANCE'])
    test_df['LOG_PURCHASES'] = np.log1p(test_df['PURCHASES'])
    
    # Scale test data
    robust_scaler = RobustScaler()
    test_scaled = robust_scaler.fit_transform(test_df)
    
    # Deserialize model results
    results_dict = pickle.loads(model_results)
    
    # Make prediction
    prediction = loaded_model.predict(test_scaled)
    
    print("\n" + "="*50)
    print("FINAL PREDICTION RESULTS")
    print("="*50)
    print(f"Best Model Used: {results_dict['best_model'].upper()}")
    print(f"Optimal Number of Clusters: {results_dict['optimal_k']}")
    print(f"Test Sample Assigned to Cluster: {prediction[0]}")
    print("="*50 + "\n")
    
    return {
        'prediction': int(prediction[0]),
        'optimal_clusters': results_dict['optimal_k'],
        'best_model': results_dict['best_model']
    }