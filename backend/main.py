from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
from sklearn.decomposition import PCA
import json
import os
from algorithms.supervised import SUPERVISED_ALGORITHMS
from algorithms.unsupervised import UNSUPERVISED_ALGORITHMS, ASSOCIATION_ALGORITHMS
from datasets import DEFAULT_DATASETS, load_dataset
from evaluation import (
    evaluate_supervised,
    evaluate_regression,
    evaluate_unsupervised,
    get_status_supervised,
    get_status_regression,
    get_status_unsupervised,
    apriori_association,
)
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
        ,
        'agglomerative': {
            'name': 'Agglomerative Clustering',
            'default_params': {'n_clusters': 3, 'linkage': 'ward'},
            'compatible_types': ['clustering'],
            'description': 'Hierarchical agglomerative clustering'
        },
        'birch': {
            'name': 'Birch',
            'default_params': {'n_clusters': 3, 'threshold': 0.5},
            'compatible_types': ['clustering'],
            'description': 'Balanced iterative reducing and clustering using hierarchies'
        }
    }
    ,
    'association': {
        'apriori': {
            'name': 'Apriori (Association Rules)',
            'default_params': {'min_support': 0.1, 'min_threshold': 0.2},
            'compatible_types': [],
            'description': 'Frequent itemset mining and association rules (transaction data)'
        },
        'fp_growth': {
            'name': 'FP-Growth (Association Rules)',
            'default_params': {'min_support': 0.1, 'min_threshold': 0.2},
            'compatible_types': [],
            'description': 'Fast frequent pattern mining using FP-growth'
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


class KaggleDatasetRequest(BaseModel):
    dataset_slug: str  # e.g., "uciml/iris"
    file_name: str    # e.g., "Iris.csv"
    target_column: str = None

@app.post("/import_kaggle")
async def import_kaggle_dataset(request: KaggleDatasetRequest):
    """Import a dataset from Kaggle using the Kaggle API."""
    try:
        import kaggle
        import os
        
        # Verify Kaggle API credentials exist
        kaggle_dir = os.path.expanduser('~/.kaggle')
        if not os.path.exists(os.path.join(kaggle_dir, 'kaggle.json')):
            raise HTTPException(
                status_code=400,
                detail="Kaggle API credentials not found. Please place your kaggle.json in ~/.kaggle/"
            )
            
        # Create data directory if it doesn't exist
        os.makedirs('user_datasets', exist_ok=True)
        
        # Download the dataset file
        try:
            kaggle.api.dataset_download_file(
                dataset=request.dataset_slug,
                file_name=request.file_name,
                path='user_datasets'
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download dataset: {str(e)}"
            )
        
        # Read the downloaded file
        file_path = os.path.join('user_datasets', request.file_name)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please use CSV or Excel files."
            )
            
        # Basic dataset info
        info = {
            'name': request.file_name,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'sample': df.head(5).to_dict(orient='records'),
            'na_count': df.isna().sum().to_dict()
        }
        
        # Suggest column roles
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
        
        # Simple heuristics for target column suggestion
        suggested_target = None
        if request.target_column and request.target_column in df.columns:
            suggested_target = request.target_column
        elif 'target' in df.columns:
            suggested_target = 'target'
        elif 'label' in df.columns:
            suggested_target = 'label'
        elif 'class' in df.columns:
            suggested_target = 'class'
            
        return {
            'success': True,
            'info': info,
            'numeric_columns': numeric_cols.tolist(),
            'categorical_columns': categorical_cols.tolist(),
            'suggested_target': suggested_target
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error importing Kaggle dataset: {str(e)}"
        )

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...), apply_cleaning: str = Form('true')):
    # Read file (support CSV and Excel)
    filename = file.filename
    if filename.endswith('.csv'):
        df = pd.read_csv(file.file)
    elif filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file.file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a CSV or Excel file.")

    # Detect transaction-style uploaded tables BEFORE applying cleaning.
    # If the file looks like a market-basket file (items across several columns, many empty cells),
    # collapse it into a single 'transaction' column and skip cleaning to preserve item labels.
    def detect_and_collapse_transactions(orig_df):
        try:
            if orig_df.shape[1] <= 1:
                return None
            non_null_per_row = orig_df.notna().sum(axis=1)
            non_null_mean = non_null_per_row.mean()
            # If average non-null items per row is small relative to columns, it's likely a transaction table
            if non_null_mean < max(6, 0.5 * orig_df.shape[1]):
                def row_to_transaction(row):
                    items = [str(v).strip() for v in row.tolist() if pd.notna(v) and str(v).strip() != '']
                    return ','.join(items)
                transactions_series = orig_df.apply(row_to_transaction, axis=1)
                return pd.DataFrame({'transaction': transactions_series})
        except Exception:
            return None
        return None

    collapsed = detect_and_collapse_transactions(df)

    # Decide whether to apply automatic cleaning based on the form field (default true).
    apply_cleaning_bool = str(apply_cleaning).lower() in ('1', 'true', 'yes', 'y')

    if collapsed is not None:
        # Use collapsed transaction column and skip cleaning to preserve item strings
        cleaned_df = collapsed
        cleaning_report = {
            'cleaning_applied': False,
            'note': 'Detected transaction-style upload; collapsed into single transaction column.'
        }
        apply_cleaning_bool = False
    else:
        if apply_cleaning_bool:
            cleaned_df, cleaning_report = clean_dataset(df)
            cleaning_report['cleaning_applied'] = True
        else:
            # Skip cleaning: keep original dataframe as-is and produce a minimal report
            cleaned_df = df.copy()
            cleaning_report = {
                'cleaning_applied': False,
                'initial_shape': df.shape,
                'final_shape': df.shape,
                'duplicates_removed': 0,
                'missing_values': {},
                'encoded_columns': [],
                'removed_columns': [],
                'type_conversions': [],
                'date_features_added': []
            }

    # Heuristic: if uploaded CSV appears to be a transaction table (many columns but few non-null per row),
    # collapse to a single-column DataFrame where each row is a comma-separated transaction string.
    try:
        if cleaned_df.shape[1] > 1:
            non_null_mean = cleaned_df.notna().sum(axis=1).mean()
            # If average non-null values per row is relatively small compared to column count,
            # treat this as a market-basket style file (items across columns)
            if non_null_mean < max(6, 0.5 * cleaned_df.shape[1]):
                # Build a single 'transaction' column by joining non-empty values per row
                def row_to_transaction(row):
                    items = [str(v).strip() for v in row.tolist() if pd.notna(v) and str(v).strip() != '']
                    return ','.join(items)
                transactions_series = cleaned_df.apply(row_to_transaction, axis=1)
                cleaned_df = pd.DataFrame({'transaction': transactions_series})
                cleaning_report['note'] = 'Detected transaction-style CSV; collapsed into single transaction column.'
    except Exception:
        # If heuristic fails for any reason, fall back to original cleaned_df
        pass

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
def get_compatible_datasets(algorithm: str):
    """Return compatible default datasets for a given algorithm"""
    if algorithm in ALGORITHMS['supervised']:
        compatible_types = ALGORITHMS['supervised'][algorithm]['compatible_types']
    elif algorithm in ALGORITHMS['unsupervised']:
        compatible_types = ALGORITHMS['unsupervised'][algorithm]['compatible_types']
    elif algorithm in ALGORITHMS['association']:
        compatible_types = ['association']
    else:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    # Build a JSON-serializable summary for each compatible dataset (exclude loader and non-serializable items)
    compatible_datasets = {}
    for name, info in DEFAULT_DATASETS.items():
        is_compatible = False
        if 'association' in compatible_types:
            is_compatible = 'association' in info.get('types', [])
        else:
            is_compatible = any(t in info.get('types', []) for t in compatible_types)

        if not is_compatible:
            continue

        compatible_datasets[name] = {
            'description': info.get('description'),
            'types': info.get('types'),
            'features': info.get('features'),
            'samples': info.get('samples'),
            'classes': info.get('classes') if 'classes' in info else None
        }

    algorithm_info = None
    for group in ALGORITHMS:
        if algorithm in ALGORITHMS[group]:
            algorithm_info = ALGORITHMS[group][algorithm]
            break

    return {
        "compatible_datasets": compatible_datasets,
        "algorithm_info": algorithm_info
    }

# @app.get("/datasets/kaggle")
# async def list_kaggle_datasets(task_type: str):
#     """List Kaggle datasets by ML task type (classification, regression, clustering)"""
#     try:
#         datasets = search_kaggle_datasets(task_type)
#         return {"datasets": datasets}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# Kaggle dataset import endpoint
class KaggleImportRequest(BaseModel):
    dataset_ref: str
    file_name: str
    target_col: str = None

# @app.post("/datasets/kaggle/import")
# async def import_kaggle_dataset(request: KaggleImportRequest):
#     """Download and preprocess a Kaggle dataset."""
#     try:
#         X, y, df = download_kaggle_dataset(request.dataset_ref, request.file_name, request.target_col)
#         # Optionally run cleaning/preprocessing here
#         # For now, just return basic info
#         return {
#             "columns": df.columns.tolist(),
#             "shape": df.shape,
#             "preview": df.head(10).to_dict(orient="records"),
#             "target_col": request.target_col,
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(request: TrainRequest):
    try:
        # Load dataset (default or user)
        if request.is_user_dataset:
            df = pd.read_csv(os.path.join('user_datasets', request.dataset_name))
            # Keep a copy of the original raw dataframe for association mining
            raw_df = df.copy()
            # Only require a target for supervised algorithms
            if request.algorithm in ALGORITHMS['supervised']:
                if not request.target_column or request.target_column not in df.columns:
                    raise HTTPException(status_code=400, detail="Target column must be specified for user dataset when using a supervised algorithm.")
                y = df[request.target_column]
                X = df.drop(columns=[request.target_column])
            else:
                # Unsupervised or association: no target; use entire dataframe as features/raw data
                y = None
                X = df.copy()
        else:
            dataset = load_dataset(request.dataset_name)
            # Convert default dataset Bunch to DataFrame when needed
            X = dataset.data
            y = getattr(dataset, 'target', None)
            # build a raw dataframe for association mining if needed
            try:
                raw_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
                if hasattr(dataset, 'target'):
                    raw_df['target'] = dataset.target
            except Exception:
                raw_df = pd.DataFrame(dataset.data)

        # raw_df should point to the original DataFrame for user datasets too
        if request.is_user_dataset:
            raw_df = df.copy()

        # Special handling for association algorithms (transaction mining)
        if request.algorithm in ASSOCIATION_ALGORITHMS:
            # Build transactions from raw_df
            def df_to_transactions(df_):
                transactions = []
                # single-column of comma-separated items or list-like
                if df_.shape[1] == 1:
                    col = df_.columns[0]
                    if df_[col].apply(lambda x: isinstance(x, (list, tuple))).all():
                        return df_[col].tolist()
                    if df_[col].dtype == object and df_[col].astype(str).str.contains(',').any():
                        return df_[col].fillna('').astype(str).apply(lambda s: [i.strip() for i in s.split(',') if i.strip()]).tolist()
                    # otherwise treat each cell as single-item transaction
                    return df_[col].fillna('').astype(str).apply(lambda s: [s] if s != '' else []).tolist()

                # Multi-column: if values are 0/1, treat columns with 1 as items
                if df_.applymap(lambda v: pd.isna(v) or v in (0, 1, True, False)).all().all():
                    for _, row in df_.iterrows():
                        items = [col for col in df_.columns if row[col] in (1, True)]
                        transactions.append(items)
                    return transactions

                # Fallback: take non-null values per row as items (stringified)
                for _, row in df_.iterrows():
                    items = [f"{col}={row[col]}" for col in df_.columns if pd.notna(row[col])]
                    transactions.append(items)
                return transactions

            transactions = df_to_transactions(raw_df)
            try:
                freq, rules = apriori_association(
                    transactions,
                    min_support=request.params.get('min_support', 0.1),
                    min_threshold=request.params.get('min_threshold', 0.2)  # Match new default
                )

                # Helper to clean item labels (strip numeric column prefixes like '0=' or '1=')
                def clean_item_label(s):
                    try:
                        s = str(s)
                        # remove leading numeric index like '0=' or '3='
                        if '=' in s and s.split('=')[0].isdigit():
                            return '='.join(s.split('=')[1:]).strip()
                        return s.strip()
                    except Exception:
                        return str(s)

                # Compute top single items counts/support even if freq is empty
                from collections import Counter
                from itertools import combinations
                flat_items = [clean_item_label(it) for t in transactions for it in t]
                counter = Counter(flat_items)
                n_transactions = len(transactions)
                top_items = [
                    {'item': k, 'count': v, 'support': v / n_transactions}
                    for k, v in counter.most_common(50)
                ]

                # Compute pair co-occurrence stats (unordered pairs) to show top associated pairs
                pair_counts = Counter()
                for t in transactions:
                    uniq = set(clean_item_label(it) for it in t if it is not None)
                    for a, b in combinations(sorted(uniq), 2):
                        pair_counts[(a, b)] += 1

                top_pairs = []
                for (a, b), cnt in pair_counts.most_common(50):
                    support = cnt / n_transactions
                    # confidences A->B and B->A
                    conf_a_b = cnt / counter[a] if counter[a] > 0 else 0
                    conf_b_a = cnt / counter[b] if counter[b] > 0 else 0
                    lift_a_b = conf_a_b / (counter[b] / n_transactions) if counter[b] > 0 else 0
                    lift_b_a = conf_b_a / (counter[a] / n_transactions) if counter[a] > 0 else 0
                    top_pairs.append({
                        'pair': [a, b],
                        'count': cnt,
                        'support': support,
                        'conf_a_b': conf_a_b,
                        'lift_a_b': lift_a_b,
                        'conf_b_a': conf_b_a,
                        'lift_b_a': lift_b_a,
                    })

                # Serialize rules and frequent itemsets to JSON-serializable structures
                freq_out = []
                if hasattr(freq, 'to_dict') and getattr(freq, 'shape', (0, 0))[0] > 0:
                    freq_df = freq.copy()
                    if 'itemsets' in freq_df.columns:
                        def clean_itemsets(s):
                            try:
                                items = list(s) if hasattr(s, '__iter__') else [s]
                                return [clean_item_label(it) for it in items]
                            except Exception:
                                return [clean_item_label(s)]
                        freq_df['itemsets'] = freq_df['itemsets'].apply(clean_itemsets)
                    freq_out = freq_df.to_dict(orient='records')

                rules_out = []
                if hasattr(rules, 'to_dict') and getattr(rules, 'shape', (0, 0))[0] > 0:
                    rules_df = rules.copy()
                    def clean_rule_side(s):
                        try:
                            items = list(s) if hasattr(s, '__iter__') else [s]
                            return [clean_item_label(it) for it in items]
                        except Exception:
                            return [clean_item_label(s)]
                    for col in ['antecedents', 'consequents']:
                        if col in rules_df.columns:
                            rules_df[col] = rules_df[col].apply(clean_rule_side)
                    rules_out = rules_df.to_dict(orient='records')

                # Compute simple evaluation metrics for association rules.
                # If explicit rules are present, use their support/confidence/lift;
                # otherwise fall back to frequent itemsets (for support) and
                # pair co-occurrence stats (for confidence/lift).
                rule_supports = [r.get('support', None) for r in rules_out] if rules_out else []
                rule_confidences = [r.get('confidence', None) for r in rules_out] if rules_out else []
                rule_lifts = [r.get('lift', None) for r in rules_out] if rules_out else []

                # Fallbacks when no explicit rules were generated
                if not rule_supports or all(v is None for v in rule_supports):
                    # Try to use supports from frequent itemsets (if available)
                    fi_supports = [fi.get('support') for fi in freq_out if isinstance(fi, dict) and fi.get('support') is not None]
                    rule_supports = fi_supports

                if not rule_confidences or all(v is None for v in rule_confidences):
                    # Use confidences computed from top_pairs if available
                    pair_confs = []
                    for p in top_pairs:
                        if 'conf_a_b' in p and p['conf_a_b'] is not None:
                            pair_confs.append(p['conf_a_b'])
                        if 'conf_b_a' in p and p['conf_b_a'] is not None:
                            pair_confs.append(p['conf_b_a'])
                    rule_confidences = pair_confs

                if not rule_lifts or all(v is None for v in rule_lifts):
                    # Use lifts computed from top_pairs if available
                    pair_lifts = []
                    for p in top_pairs:
                        if 'lift_a_b' in p and p['lift_a_b'] is not None:
                            pair_lifts.append(p['lift_a_b'])
                        if 'lift_b_a' in p and p['lift_b_a'] is not None:
                            pair_lifts.append(p['lift_b_a'])
                    rule_lifts = pair_lifts

                # Clean lists (remove None) before computing means
                rule_supports_clean = [float(v) for v in rule_supports if v is not None]
                rule_confidences_clean = [float(v) for v in rule_confidences if v is not None]
                rule_lifts_clean = [float(v) for v in rule_lifts if v is not None]

                eval_summary = {
                    'n_rules': len(rules_out),
                    'n_frequent_itemsets': len(freq_out),
                    'avg_support': float(np.mean(rule_supports_clean)) if rule_supports_clean else None,
                    'avg_confidence': float(np.mean(rule_confidences_clean)) if rule_confidences_clean else None,
                    'avg_lift': float(np.mean(rule_lifts_clean)) if rule_lifts_clean else None
                }

                # Save rules + frequent itemsets as a model artifact (JSON) for download
                os.makedirs('models', exist_ok=True)
                safe_dataset = request.dataset_name.replace('/', '_').replace('\\', '_') if request.dataset_name else 'user'
                model_filename = f"models/{safe_dataset}_{request.algorithm}_rules.json"
                try:
                    with open(model_filename, 'w', encoding='utf-8') as mf:
                        json.dump({
                            'dataset_name': request.dataset_name,
                            'algorithm': request.algorithm,
                            'n_transactions': n_transactions,
                            'frequent_itemsets': freq_out,
                            'rules': rules_out,
                            'top_items': top_items,
                            'top_pairs': top_pairs,
                            'evaluation': eval_summary
                        }, mf, ensure_ascii=False, indent=2)
                except Exception:
                    model_filename = None

                return {
                    'algorithm': request.algorithm,
                    'n_transactions': n_transactions,
                    'frequent_itemsets': freq_out,
                    'rules': rules_out,
                    'top_items': top_items,
                    'top_pairs': top_pairs,
                    'n_frequent_itemsets': len(freq_out),
                    'n_rules': len(rules_out),
                    'dataset_name': request.dataset_name,
                    'status': 'n/a',
                    'evaluation': eval_summary,
                    'model_path': model_filename
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Preprocessing: handle missing values, encode categoricals, scale numerics for other algorithms
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
    # Support multiple model artifact formats: .pkl, .joblib, .json
    models_dir = 'models'
    candidates = [
        os.path.join(models_dir, f"{dataset_name}_{algorithm}.pkl"),
        os.path.join(models_dir, f"{dataset_name}_{algorithm}.joblib"),
        os.path.join(models_dir, f"{dataset_name}_{algorithm}_rules.json"),
        os.path.join(models_dir, f"{dataset_name}_{algorithm}_model.pkl"),
        os.path.join(models_dir, f"cluster_{dataset_name}_{algorithm}.joblib")
    ]
    found = None
    for c in candidates:
        if c and os.path.exists(c):
            found = c
            break
    if not found:
        # try any file that starts with dataset_name_algorithm
        for f in os.listdir(models_dir) if os.path.exists(models_dir) else []:
            if f.startswith(f"{dataset_name}_{algorithm}") or f.startswith(f"cluster_{dataset_name}_{algorithm}"):
                found = os.path.join(models_dir, f)
                break
    if not found:
        raise HTTPException(status_code=404, detail="Model not found")

    filename = os.path.basename(found)
    return FileResponse(found, media_type='application/octet-stream', filename=filename)


@app.get("/download_model")
async def download_latest_model():
    """Download the most recently saved clustering model (by filename sorting)."""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        raise HTTPException(status_code=404, detail="No models available")
    files = [f for f in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, f))]
    if not files:
        raise HTTPException(status_code=404, detail="No models found")
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(models_dir, f)))
    path = os.path.join(models_dir, latest)
    return FileResponse(path, media_type='application/octet-stream', filename=latest)


class ClusterRequest(BaseModel):
    dataset_name: str
    is_user_dataset: bool = False
    feature_columns: list = None
    max_k: int = 10
    n_clusters: int = 3
    params: dict = {}


@app.post("/cluster/elbow")
async def cluster_elbow_endpoint(req: ClusterRequest):
    """Return inertia values for k=2..max_k to support elbow plots."""
    try:
        # Load dataset
        if req.is_user_dataset:
            df = pd.read_csv(os.path.join('user_datasets', req.dataset_name))
            data_df = df.copy()
        else:
            dataset = load_dataset(req.dataset_name)
            try:
                data_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            except Exception:
                data_df = pd.DataFrame(dataset.data)

        # Select feature columns if provided
        if req.feature_columns:
            data_df = data_df[req.feature_columns]

        # Preprocess: fill missing, encode categoricals, scale numeric
        X = data_df.copy()
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "")
            else:
                X[col] = X[col].fillna(X[col].mean())
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.factorize(X[col])[0]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Compute elbow
        from evaluation import cluster_elbow as cluster_elbow_fn
        results = cluster_elbow_fn(
            X_scaled, 
            k_range=range(2, min(req.max_k, 50) + 1),
            dataset_name=req.dataset_name if req.is_user_dataset else None
        )
        return {
            "elbow": results,
            "columns": list(data_df.columns)  # Return columns used for clustering
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cluster/metrics")
async def cluster_metrics_endpoint(req: ClusterRequest):
    """Run clustering with specified algorithm and n_clusters, return metrics."""
    try:
        # Load dataset
        if req.is_user_dataset:
            df = pd.read_csv(os.path.join('user_datasets', req.dataset_name))
            data_df = df.copy()
        else:
            dataset = load_dataset(req.dataset_name)
            try:
                data_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            except Exception:
                data_df = pd.DataFrame(dataset.data)

        # Select feature columns if provided
        if req.feature_columns:
            data_df = data_df[req.feature_columns]

        # Preprocess: fill missing, encode categoricals, scale numeric
        X = data_df.copy()
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "")
            else:
                X[col] = X[col].fillna(X[col].mean())
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.factorize(X[col])[0]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Choose algorithm class from mapping; default to kmeans
        alg = req.params.get('algorithm', 'kmeans')
        if alg not in UNSUPERVISED_ALGORITHMS:
            raise HTTPException(status_code=400, detail=f"Unknown unsupervised algorithm: {alg}")
        model_cls = UNSUPERVISED_ALGORITHMS[alg]
        model_params = req.params.get('model_params', {})
        # ensure n_clusters param if applicable
        if 'n_clusters' not in model_params and hasattr(model_cls, 'n_clusters') or alg in ('kmeans','agglomerative','birch'):
            model_params['n_clusters'] = req.n_clusters

        model = model_cls(**model_params)
        # fit and get labels
        if hasattr(model, 'fit_predict'):
            labels = model.fit_predict(X_scaled)
        else:
            model.fit(X_scaled)
            labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_scaled)

        from evaluation import cluster_metrics as cluster_metrics_fn
        metrics = cluster_metrics_fn(X_scaled, labels, model=model)

        # Save model for download (filename includes dataset and algorithm)
        os.makedirs('models', exist_ok=True)
        safe_dataset = req.dataset_name.replace('/', '_').replace('\\', '_') if req.dataset_name else 'user'
        model_path = f"models/cluster_{safe_dataset}_{alg}_{int(req.n_clusters)}.joblib"
        try:
            joblib.dump(model, model_path)
        except Exception:
            # non-fatal; continue without failing if model can't be saved
            model_path = None

        return {
            'algorithm': alg,
            'n_clusters': int(req.n_clusters),
            'metrics': metrics,
            'n_samples': X.shape[0],
            'model_path': model_path
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cluster/visualize")
async def cluster_visualize_endpoint(req: ClusterRequest):
    """Fit clustering model and return 2D PCA projection of points and centroids for visualization."""
    try:
        # Load dataset
        if req.is_user_dataset:
            df = pd.read_csv(os.path.join('user_datasets', req.dataset_name), sep=';')
            data_df = df.copy()
        else:
            from sklearn.datasets import load_iris, load_wine, load_digits
            dataset_map = {'iris': load_iris(), 'wine': load_wine(), 'digits': load_digits()}
            if req.dataset_name not in dataset_map:
                raise HTTPException(status_code=400, detail="Dataset not found")
            dataset = dataset_map[req.dataset_name]
            try:
                data_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            except Exception:
                data_df = pd.DataFrame(dataset.data)

        if req.feature_columns:
            data_df = data_df[req.feature_columns]

        # Preprocess
        X = data_df.copy()
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "")
            else:
                X[col] = X[col].fillna(X[col].mean())
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.factorize(X[col])[0]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        alg = req.params.get('algorithm', 'kmeans')
        if alg not in UNSUPERVISED_ALGORITHMS:
            raise HTTPException(status_code=400, detail=f"Unknown unsupervised algorithm: {alg}")
        model_cls = UNSUPERVISED_ALGORITHMS[alg]
        model_params = req.params.get('model_params', {})
        if 'n_clusters' not in model_params and hasattr(model_cls, 'n_clusters') or alg in ('kmeans','agglomerative','birch'):
            model_params['n_clusters'] = req.n_clusters

        model = model_cls(**model_params)
        if hasattr(model, 'fit_predict'):
            labels = model.fit_predict(X_scaled)
        else:
            model.fit(X_scaled)
            labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_scaled)

        # PCA for 2D projection
        if X_scaled.shape[1] >= 2:
            pca = PCA(n_components=2)
            X2 = pca.fit_transform(X_scaled)
            centroids = []
            if hasattr(model, 'cluster_centers_'):
                try:
                    centroids = pca.transform(model.cluster_centers_).tolist()
                except Exception:
                    centroids = []
            explained_variance = pca.explained_variance_ratio_.tolist()
        else:
            # fallback for 1D data
            X2 = np.hstack([X_scaled.reshape(-1,1), np.random.normal(0,0.05,size=(X_scaled.shape[0],1))])
            centroids = [[float(c),0.0] for c in getattr(model,'cluster_centers_',[])]
            explained_variance = [1.0,0.0]

        # Color palette (more distinct for better quality)
        labels_arr = np.array(labels)
        unique_labels = np.unique(labels_arr)
        palette = [
            'rgba(255,99,132,0.8)','rgba(54,162,235,0.8)','rgba(255,206,86,0.8)','rgba(75,192,192,0.8)',
            'rgba(153,102,255,0.8)','rgba(255,159,64,0.8)','rgba(99,255,132,0.8)','rgba(66,135,245,0.8)',
            'rgba(255,0,255,0.8)','rgba(0,255,255,0.8)','rgba(255,128,0,0.8)'
        ]
        colors = [palette[int(l) % len(palette)] for l in labels_arr]

        points = [{'x': float(x), 'y': float(y), 'size': 8, 'opacity':0.7} for x, y in X2]
        centroids_json = [{'x': float(c[0]), 'y': float(c[1]), 'size': 12, 'opacity':1.0} for c in centroids] if centroids else []

        # Save model
        os.makedirs('models', exist_ok=True)
        safe_dataset = req.dataset_name.replace('/', '_').replace('\\', '_') if req.dataset_name else 'user'
        model_path = f"models/cluster_{safe_dataset}_{alg}_{int(req.n_clusters)}.joblib"
        try:
            joblib.dump(model, model_path)
        except Exception:
            model_path = None

        return {
            'points': points,
            'centroids': centroids_json,
            'labels': labels_arr.tolist(),
            'colors': colors,
            'explained_variance': explained_variance,
            'model_path': model_path
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))