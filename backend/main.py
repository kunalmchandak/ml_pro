from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from algorithms.supervised import SUPERVISED_ALGORITHMS
from algorithms.unsupervised import UNSUPERVISED_ALGORITHMS
from datasets import DEFAULT_DATASETS, load_dataset
from evaluation import evaluate_supervised, evaluate_regression, evaluate_unsupervised, get_status_supervised, get_status_regression, get_status_unsupervised
from fastapi.responses import FileResponse

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Update ALGORITHMS dictionary to use strings instead of class references
ALGORITHMS = {
    'supervised': {
        'logistic_regression': {
            'name': 'Logistic Regression',
            'task': 'classification',
            'default_params': {'max_iter': 1000},
            'compatible_types': ['classification', 'binary'],
            'description': 'Best for binary/multiclass classification'
        },
        'random_forest': {
            'name': 'Random Forest',
            'task': 'classification',
            'default_params': {'n_estimators': 100},
            'compatible_types': ['classification', 'binary'],
            'description': 'Versatile classifier for all types'
        },
        'svm': {
            'name': 'Support Vector Machine',
            'task': 'classification',
            'default_params': {'kernel': 'rbf'},
            'compatible_types': ['classification', 'binary'],
            'description': 'Good for high-dimensional data'
        },
        'knn': {
            'name': 'K-Nearest Neighbors',
            'task': 'classification',
            'default_params': {'n_neighbors': 5},
            'compatible_types': ['classification', 'binary'],
            'description': 'Instance-based learning algorithm'
        },
        'decision_tree': {
            'name': 'Decision Tree',
            'task': 'classification',
            'default_params': {'max_depth': None},
            'compatible_types': ['classification', 'binary'],
            'description': 'Tree-based model for classification'
        },
        'linear_regression': {
            'name': 'Linear Regression',
            'task': 'regression',
            'default_params': {},
            'compatible_types': ['regression'],
            'description': 'Linear model for continuous target prediction'
        },
        'ridge': {
            'name': 'Ridge Regression',
            'task': 'regression',
            'default_params': {'alpha': 1.0},
            'compatible_types': ['regression'],
            'description': 'Ridge regression with L2 regularization'
        },
        'lasso': {
            'name': 'Lasso Regression',
            'task': 'regression',
            'default_params': {'alpha': 1.0},
            'compatible_types': ['regression'],
            'description': 'Lasso regression with L1 regularization'
        },
        'decision_tree_regressor': {
            'name': 'Decision Tree Regressor',
            'task': 'regression',
            'default_params': {'max_depth': None},
            'compatible_types': ['regression'],
            'description': 'Tree-based regression model'
        },
        'random_forest_regressor': {
            'name': 'Random Forest Regressor',
            'task': 'regression',
            'default_params': {'n_estimators': 100},
            'compatible_types': ['regression'],
            'description': 'Ensemble of decision trees for regression'
        },
        'gradient_boosting': {
            'name': 'Gradient Boosting Regressor',
            'task': 'regression',
            'default_params': {'n_estimators': 100},
            'compatible_types': ['regression'],
            'description': 'Gradient boosting for regression'
        },
        'xgboost': {
            'name': 'XGBoost Regressor',
            'task': 'regression',
            'default_params': {'n_estimators': 100},
            'compatible_types': ['regression'],
            'description': 'Extreme gradient boosting for regression'
        },
        'lightgbm': {
            'name': 'LightGBM Regressor',
            'task': 'regression',
            'default_params': {'n_estimators': 100},
            'compatible_types': ['regression'],
            'description': 'Light gradient boosting for regression'
        },
        'svr': {
            'name': 'Support Vector Regressor',
            'task': 'regression',
            'default_params': {'kernel': 'rbf'},
            'compatible_types': ['regression'],
            'description': 'Support vector machine for regression'
        },
        'knn_regressor': {
            'name': 'K-Nearest Neighbors Regressor',
            'task': 'regression',
            'default_params': {'n_neighbors': 5},
            'compatible_types': ['regression'],
            'description': 'Instance-based regression'
        }
    },
    'unsupervised': {
        'kmeans': {
            'name': 'K-Means Clustering',
            'default_params': {'n_clusters': 3, 'random_state': 42},
            'compatible_types': ['clustering'],
            'description': 'Clustering algorithm for unlabeled data'
        },
        'dbscan': {
            'name': 'DBSCAN',
            'default_params': {'eps': 1.9, 'min_samples': 4},
            'compatible_types': ['clustering'],
            'description': 'Density-based clustering algorithm'
        }
    }
}

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

class TrainRequest(BaseModel):
    algorithm: str
    dataset_name: str
    params: dict = {}

@app.get("/datasets/default")
async def list_default_datasets():
    """List all available default datasets"""
    return {
        name: {
            'description': info['description'],
            'type': info['type']
        }
        for name, info in DEFAULT_DATASETS.items()
    }

@app.get("/datasets/default/{dataset_name}")
async def get_default_dataset(dataset_name: str):
    """Load a specific default dataset"""
    if dataset_name not in DEFAULT_DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = DEFAULT_DATASETS[dataset_name]['loader']()
    df = pd.DataFrame(
        data=dataset.data,
        columns=dataset.feature_names
    )
    
    # Add target column for supervised learning datasets
    if hasattr(dataset, 'target'):
        df['target'] = dataset.target
    
    return {
        "name": dataset_name,
        "description": DEFAULT_DATASETS[dataset_name]['description'],
        "data": df.to_dict(orient='records'),
        "columns": df.columns.tolist(),
        "shape": df.shape
    }

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    return {"message": "Dataset uploaded successfully", "columns": df.columns.tolist()}

# Helper function to get model class
def get_model_class(algorithm: str):
    if algorithm in SUPERVISED_ALGORITHMS:
        return SUPERVISED_ALGORITHMS[algorithm]
    elif algorithm in UNSUPERVISED_ALGORITHMS:
        return UNSUPERVISED_ALGORITHMS[algorithm]
    raise ValueError(f"Unknown algorithm: {algorithm}")

@app.get("/datasets/compatible/{algorithm}")
async def get_compatible_datasets(algorithm: str):
    """Get datasets compatible with selected algorithm"""
    for category in ALGORITHMS:
        if algorithm in ALGORITHMS[category]:
            algo_info = ALGORITHMS[category][algorithm]
            compatible_types = algo_info['compatible_types']
            
            compatible_datasets = {
                name: {
                    'description': info['description'],
                    'features': info['features'],
                    'samples': info['samples'],
                    'types': info['types']
                }
                for name, info in DEFAULT_DATASETS.items()
                if any(t in compatible_types for t in info['types'])
            }
            
            return {
                "algorithm_info": {
                    "name": algo_info['name'],
                    "description": algo_info['description'],
                    "default_params": algo_info['default_params']
                },
                "compatible_datasets": compatible_datasets
            }
    
    raise HTTPException(status_code=404, detail="Algorithm not found")

@app.post("/train")
async def train_model(request: TrainRequest):
    try:
        # Load dataset
        dataset = load_dataset(request.dataset_name)
        X = dataset.data

        # Preprocess features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Get model class
        model_class = get_model_class(request.algorithm)

        # Determine if supervised or unsupervised
        is_supervised = request.algorithm in ALGORITHMS['supervised']

        # Get default parameters
        category = 'supervised' if is_supervised else 'unsupervised'
        default_params = ALGORITHMS[category][request.algorithm]['default_params']
        params = {**default_params, **request.params}

        if is_supervised:
            # Supervised learning
            y = dataset.target
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            model = model_class(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            algo_info = ALGORITHMS['supervised'][request.algorithm]
            if algo_info['task'] == 'classification':
                metrics = evaluate_supervised(y_test, y_pred)
                status = get_status_supervised(metrics)
            else:
                metrics = evaluate_regression(y_test, y_pred)
                status = get_status_regression(metrics)
        else:
            # Unsupervised learning
            model = model_class(**params)
            labels = model.fit_predict(X_scaled)
            metrics = evaluate_unsupervised(X_scaled, labels, model)
            status = get_status_unsupervised(metrics)

        # Save model and scaler
        model_path = f"models/{request.dataset_name}_{request.algorithm}.pkl"
        joblib.dump({'model': model, 'scaler': scaler}, model_path)

        # Add common metrics
        metrics.update({
            "dataset_name": request.dataset_name,
            "algorithm": request.algorithm,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "model_path": model_path,
            "status": status
        })

        return metrics

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_model/{dataset_name}/{algorithm}")
async def download_model(dataset_name: str, algorithm: str):
    model_path = f"models/{dataset_name}_{algorithm}.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename=f"{dataset_name}_{algorithm}_model.pkl"
    )
