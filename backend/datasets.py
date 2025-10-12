from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_diabetes, fetch_california_housing
from sklearn.datasets import fetch_openml
import pandas as pd

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
    },
    'california_housing': {
        'loader': fetch_california_housing,
        'description': 'Predict median house prices using census data',
        'types': ['regression'],
        'features': 8,
        'samples': 20640
    },
    'boston_housing': {
        'loader': lambda: load_url_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', target_col='medv'),
        'description': 'Predict house prices',
        'types': ['regression'],
        'features': 13,
        'samples': 506
    },
    'concrete_compressive_strength': {
        'loader': lambda: load_url_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls', target_col='Concrete compressive strength(MPa, megapascals) '),
        'description': 'Predict concrete strength based on composition',
        'types': ['regression'],
        'features': 8,
        'samples': 1030
    },
    'energy_efficiency': {
        'loader': lambda: load_url_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx', target_col='Y1'),
        'description': 'Predict heating/cooling loads of buildings',
        'types': ['regression'],
        'features': 8,
        'samples': 768
    }
}

def load_url_csv(url, target_col, sep=None):
    try:
        # Add a timeout and user agent to avoid connection issues
        headers = {'User-Agent': 'Mozilla/5.0'}
        if sep:
            df = pd.read_csv(url, sep=sep, timeout=10)
        else:
            df = pd.read_csv(url, timeout=10)
            
        # Handle missing values more gracefully
        df = df.dropna(subset=[target_col])
        
        # Convert all columns to numeric where possible
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                continue
                
        df = df.dropna()
        X = df.drop(target_col, axis=1).values
        y = df[target_col].values
        feature_names = df.drop(target_col, axis=1).columns.tolist()
        
        class Bunch:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        return Bunch(data=X, target=y, feature_names=feature_names)
        
    except Exception as e:
        print(f"Error loading dataset from {url}: {str(e)}")
        raise ValueError(f"Failed to load CSV from {url}: {e}")
    
    class Bunch:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    return Bunch(data=X, target=y, feature_names=feature_names)

def load_url_excel(url, target_col):
    try:
        df = pd.read_excel(url)
        df = df.dropna()
        X = df.drop(target_col, axis=1).values
        y = df[target_col].values
        feature_names = df.drop(target_col, axis=1).columns.tolist()
    except Exception as e:
        raise ValueError(f"Failed to load Excel from {url}: {e}")

    class Bunch:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    return Bunch(data=X, target=y, feature_names=feature_names)

def load_kaggle_csv(dataset_slug, file_name, target_col):
    try:
        import kaggle
        import os
        if not os.path.exists('data'):
            os.makedirs('data')
        kaggle.api.dataset_download_file(dataset_slug, file_name, path='data')
        df = pd.read_csv(f'data/{file_name}')
        df = df.dropna()
        X = df.drop(target_col, axis=1).values
        y = df[target_col].values
        feature_names = df.drop(target_col, axis=1).columns.tolist()
    except Exception as e:
        raise ValueError(f"Failed to load from Kaggle {dataset_slug}/{file_name}: {e}")

    class Bunch:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    return Bunch(data=X, target=y, feature_names=feature_names)

def load_dataset(dataset_name):
    if dataset_name not in DEFAULT_DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found")
    return DEFAULT_DATASETS[dataset_name]['loader']()
