from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_diabetes, fetch_california_housing, make_regression
from sklearn.datasets import fetch_openml
import pandas as pd
import requests
from io import StringIO

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
        'loader': lambda: load_url_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', 
                                     target_col='MEDV', 
                                     sep=r'\s+',
                                     names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                                           'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']),
        'description': 'Boston Housing dataset from UCI ML repository',
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
    ,
    'groceries_association': {
        'loader': lambda: load_groceries_transactions('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv'),
        'description': 'Groceries transactions for association rule mining',
        'types': ['association'],
        'features': None,
        'samples': 9835
    },
    'retail_market_basket': {
        'loader': lambda: load_market_basket_transactions([
            'https://raw.githubusercontent.com/selva86/datasets/master/Market_Basket_Optimisation.csv',
            'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv'
        ]),
        'description': 'Retail market basket transactions',
        'types': ['association'],
        'features': None,
        'samples': 7501
    }
}

def load_groceries_transactions(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        # Each line is a transaction: items separated by commas
        transactions = [line.strip().split(',') for line in response.text.splitlines() if line.strip()]
        class Bunch:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        return Bunch(data=transactions, target=None, feature_names=None)
    except Exception as e:
        print(f"Error loading groceries transactions from {url}: {str(e)}")
        raise ValueError(f"Failed to load groceries transactions from {url}: {e}")

def load_market_basket_transactions(url):
    # url may be a single string or a list of fallback URLs
    urls = url if isinstance(url, (list, tuple)) else [url]
    last_exc = None
    for u in urls:
        try:
            response = requests.get(u)
            response.raise_for_status()
            text = response.text
            # Parse the file line-by-line to handle both single-column comma-separated
            # transaction files and multi-column CSVs without relying on pandas tokenization.
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            transactions = []
            for ln in lines:
                # If the line contains commas, treat it as comma-separated items
                if ',' in ln:
                    items = [it.strip() for it in ln.split(',') if it.strip()]
                    transactions.append(items)
                    continue
                # If the line contains semicolons, split by semicolon
                if ';' in ln:
                    items = [it.strip() for it in ln.split(';') if it.strip()]
                    transactions.append(items)
                    continue
                # Otherwise, try splitting by whitespace (fallback)
                parts = [p.strip() for p in ln.split() if p.strip()]
                if len(parts) > 1:
                    transactions.append(parts)
                    continue
                # If it's a single token, keep as single-item transaction
                transactions.append([ln])

            class Bunch:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

            return Bunch(data=transactions, target=None, feature_names=None)
        except Exception as e:
            last_exc = e
            # try next URL in list
            continue
    # if none succeeded
    print(f"Error loading market basket transactions from provided URLs. Last error: {last_exc}")
    raise ValueError(f"Failed to load market basket transactions: {last_exc}")

def load_url_csv(url, target_col, sep=None, names=None):
    try:
        response = requests.get(url)
        response.raise_for_status()
        if sep:
            df = pd.read_csv(StringIO(response.text), sep=sep, names=names)
        else:
            df = pd.read_csv(StringIO(response.text), names=names)

        # For association datasets, target_col may be None
        if target_col is not None:
            df = df.dropna(subset=[target_col])
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    continue
            df = df.dropna()
            X = df.drop(target_col, axis=1).values
            y = df[target_col].values
            feature_names = df.drop(target_col, axis=1).columns.tolist()
        else:
            # For association rule mining, return raw transaction data
            X = df.values
            y = None
            feature_names = df.columns.tolist()

        class Bunch:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        return Bunch(data=X, target=y, feature_names=feature_names)
    except Exception as e:
        print(f"Error loading dataset from {url}: {str(e)}")
        raise ValueError(f"Failed to load CSV from {url}: {e}")

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
