from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_diabetes
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update DEFAULT_DATASETS with more specific type information
DEFAULT_DATASETS = {
    'iris': {
        'loader': load_iris,
        'description': 'Iris flower dataset',
        'types': ['classification', 'clustering'],
        'features': 4,
        'samples': 150,
        'classes': 3
    },
    'breast_cancer': {
        'loader': load_breast_cancer,
        'description': 'Breast cancer wisconsin dataset',
        'types': ['classification', 'binary'],
        'features': 30,
        'samples': 569,
        'classes': 2
    },
    'wine': {
        'loader': load_wine,
        'description': 'Wine recognition dataset',
        'types': ['classification', 'clustering'],
        'features': 13,
        'samples': 178,
        'classes': 3
    },
    'diabetes': {
        'loader': load_diabetes,
        'description': 'Diabetes dataset',
        'types': ['regression'],
        'features': 10,
        'samples': 442
    }
}

# Update ALGORITHMS dictionary to use strings instead of class references
ALGORITHMS = {
    'supervised': {
        'logistic_regression': {
            'name': 'Logistic Regression',
            'default_params': {'max_iter': 1000},
            'compatible_types': ['classification', 'binary'],
            'description': 'Best for binary/multiclass classification'
        },
        'random_forest': {
            'name': 'Random Forest',
            'default_params': {'n_estimators': 100},
            'compatible_types': ['classification', 'binary'],
            'description': 'Versatile classifier for all types'
        },
        'svm': {
            'name': 'Support Vector Machine',
            'default_params': {'kernel': 'rbf'},
            'compatible_types': ['classification', 'binary'],
            'description': 'Good for high-dimensional data'
        }
    },
    'unsupervised': {
        'kmeans': {
            'name': 'K-Means Clustering',
            'default_params': {'n_clusters': 3},
            'compatible_types': ['clustering'],
            'description': 'Clustering algorithm for unlabeled data'
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
    if algorithm == 'logistic_regression':
        return LogisticRegression
    elif algorithm == 'random_forest':
        return RandomForestClassifier
    elif algorithm == 'svm':
        return SVC
    elif algorithm == 'kmeans':
        return KMeans
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
        # Load and preprocess data
        if request.dataset_name in DEFAULT_DATASETS:
            dataset = DEFAULT_DATASETS[request.dataset_name]['loader']()
            X = dataset.data
            y = dataset.target
        else:
            raise HTTPException(status_code=400, detail="Dataset not found")

        # Preprocess features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Get model class using helper function
        model_class = get_model_class(request.algorithm)
        
        # Get default parameters
        for category in ALGORITHMS:
            if request.algorithm in ALGORITHMS[category]:
                default_params = ALGORITHMS[category][request.algorithm]['default_params']
                params = {**default_params, **request.params}
                
                # Train model
                model = model_class(**params)
                model.fit(X_train, y_train)
                
                # Save model and scaler
                model_path = f"models/{request.dataset_name}_{request.algorithm}.pkl"
                joblib.dump({'model': model, 'scaler': scaler}, model_path)
                
                # Get metrics
                metrics = {
                    "accuracy": model.score(X_test, y_test),
                    "dataset_name": request.dataset_name,
                    "algorithm": request.algorithm,
                    "n_samples": len(X),
                    "n_features": X.shape[1],
                    "model_path": model_path
                }
                
                return metrics
                
        raise HTTPException(status_code=400, detail="Algorithm not supported")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_model/{dataset_name}/{algorithm}")
async def download_model(dataset_name: str, algorithm: str):
    model_path = f"models/{dataset_name}_{algorithm}.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    return {"model_path": model_path}
