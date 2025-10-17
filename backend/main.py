from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import json
import os
from algorithms.supervised import SUPERVISED_ALGORITHMS
from algorithms.unsupervised import UNSUPERVISED_ALGORITHMS
from datasets import DEFAULT_DATASETS, load_dataset
from evaluation import evaluate_supervised, evaluate_regression, evaluate_unsupervised, get_status_supervised, get_status_regression, get_status_unsupervised
from fastapi.responses import FileResponse
from cleaning import clean_dataset

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
        'gaussian_nb': {
            'name': 'Gaussian Naive Bayes',
            'task': 'classification',
            'default_params': {},
            'compatible_types': ['classification', 'binary'],
            'description': 'Probabilistic classifier based on Bayes theorem with Gaussian distribution assumption'
        },
        'multinomial_nb': {
            'name': 'Multinomial Naive Bayes',
            'task': 'classification',
            'default_params': {'alpha': 1.0},
            'compatible_types': ['classification', 'binary'],
            'description': 'Naive Bayes for multinomial distributed data, good for text classification'
        },
        'bernoulli_nb': {
            'name': 'Bernoulli Naive Bayes',
            'task': 'classification',
            'default_params': {'alpha': 1.0},
            'compatible_types': ['classification', 'binary'],
            'description': 'Naive Bayes for binary/boolean features'
        },
        'gradient_boosting_classifier': {
            'name': 'Gradient Boosting Classifier',
            'task': 'classification',
            'default_params': {'n_estimators': 100},
            'compatible_types': ['classification', 'binary'],
            'description': 'Powerful boosting algorithm for classification tasks'
        },
        'xgboost_classifier': {
            'name': 'XGBoost Classifier',
            'task': 'classification',
            'default_params': {'n_estimators': 100},
            'compatible_types': ['classification', 'binary'],
            'description': 'High-performance implementation of gradient boosting'
        },
        'lightgbm_classifier': {
            'name': 'LightGBM Classifier',
            'task': 'classification',
            'default_params': {'n_estimators': 100},
            'compatible_types': ['classification', 'binary'],
            'description': 'Light and fast implementation of gradient boosting'
        },
        'catboost_classifier': {
            'name': 'CatBoost Classifier',
            'task': 'classification',
            'default_params': {'n_estimators': 100},
            'compatible_types': ['classification', 'binary'],
            'description': 'High-performance gradient boosting on decision trees'
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


# Extend TrainRequest to support user datasets and custom target
class TrainRequest(BaseModel):
    algorithm: str
    dataset_name: str  # can be default or user dataset filename
    params: dict = {}
    is_user_dataset: bool = False
    target_column: str = None

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
    # Read file (support CSV and Excel)
    filename = file.filename
    if filename.endswith('.csv'):
        df = pd.read_csv(file.file)
    elif filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file.file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a CSV or Excel file.")

    # Clean the dataset automatically
    cleaned_df, cleaning_report = clean_dataset(df)

    # Save both original and cleaned datasets
    os.makedirs('user_datasets', exist_ok=True)
    original_path = os.path.join('user_datasets', f"original_{filename}")
    cleaned_path = os.path.join('user_datasets', filename)
    df.to_csv(original_path, index=False)
    cleaned_df.to_csv(cleaned_path, index=False)

    # Get column information for all columns
    column_info = {}
    for col in cleaned_df.columns:
        nunique = cleaned_df[col].nunique()
        col_dtype = cleaned_df[col].dtype
        is_numeric = np.issubdtype(col_dtype, np.number)
        
        col_info = {
            'unique_values': int(nunique),
            'dtype': str(col_dtype),
            'is_numeric': is_numeric,
            'sample_values': cleaned_df[col].dropna().head(5).tolist(),
            'suggested_task': [],
            'suggested_algorithms': []
        }
        
        # Suggest possible ML tasks and algorithms based on column properties
        if nunique <= 20 and nunique > 1:
            col_info['suggested_task'].append('classification')
            # Suggest classification algorithms
            if nunique == 2:
                col_info['suggested_algorithms'].extend([
                    'logistic_regression',
                    'random_forest',
                    'svm',
                    'gradient_boosting_classifier'
                ])
            else:
                col_info['suggested_algorithms'].extend([
                    'random_forest',
                    'gradient_boosting_classifier',
                    'multinomial_nb',
                    'xgboost_classifier'
                ])
        
        if is_numeric and nunique > 20:
            col_info['suggested_task'].append('regression')
            # Suggest regression algorithms
            col_info['suggested_algorithms'].extend([
                'linear_regression',
                'random_forest_regressor',
                'gradient_boosting_regressor',
                'svr'
            ])
            
        column_info[col] = col_info

    # Include all columns as candidates, with their properties
    target_candidates = [col for col in cleaned_df.columns]
    
    # No specific inferred type until user selects target
    inferred_type = 'unknown'
    target_info = {}

    return {
        "filename": filename,
        "shape": cleaned_df.shape,
        "columns": cleaned_df.columns.tolist(),
        "column_info": column_info,  # Detailed information about each column
        "target_candidates": target_candidates,
        "cleaning_report": cleaning_report,
        "preview_data": cleaned_df.head(5).to_dict(orient='records')
    }

    preprocessing = {
        'missing_values': missing,
        'categorical_columns': categorical,
        'numeric_columns': numeric,
        'suggested_scaler': 'StandardScaler' if inferred_type != 'classification' or len(categorical) == 0 else 'OneHotEncoder+StandardScaler',
        'suggested_encoding': 'OneHotEncoder' if categorical else None
    }

    return {
        "message": "Dataset uploaded and analyzed successfully",
        "columns": df.columns.tolist(),
        "shape": df.shape,
        "target_candidates": target_candidates,
        "inferred_type": inferred_type,
        "target_info": target_info,
        "preprocessing": preprocessing,
        "save_path": save_path
    }

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

@app.get("/datasets/kaggle")
async def list_kaggle_datasets(task_type: str):
    """List Kaggle datasets by ML task type (classification, regression, clustering)"""
    try:
        datasets = search_kaggle_datasets(task_type)
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Kaggle dataset import endpoint
class KaggleImportRequest(BaseModel):
    dataset_ref: str
    file_name: str
    target_col: str = None

@app.post("/datasets/kaggle/import")
async def import_kaggle_dataset(request: KaggleImportRequest):
    """Download and preprocess a Kaggle dataset."""
    try:
        X, y, df = download_kaggle_dataset(request.dataset_ref, request.file_name, request.target_col)
        # Optionally run cleaning/preprocessing here
        # For now, just return basic info
        return {
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "preview": df.head(10).to_dict(orient="records"),
            "target_col": request.target_col,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(request: TrainRequest):
    try:
        # Load dataset (default or user)
        if request.is_user_dataset:
            df = pd.read_csv(os.path.join('user_datasets', request.dataset_name))
            # Target column must be specified for user datasets
            if not request.target_column or request.target_column not in df.columns:
                raise HTTPException(status_code=400, detail="Target column must be specified for user dataset.")
            y = df[request.target_column]
            X = df.drop(columns=[request.target_column])
        else:
            dataset = load_dataset(request.dataset_name)
            X = dataset.data
            y = getattr(dataset, 'target', None)

        # Preprocessing: handle missing values, encode categoricals, scale numerics
        X = pd.DataFrame(X)
        # Fill missing values
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "")
            else:
                X[col] = X[col].fillna(X[col].mean())
        # Encode categoricals
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.factorize(X[col])[0]
        # Scale features
        if request.algorithm == 'multinomial_nb':
            scaler = MinMaxScaler()
        else:
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
