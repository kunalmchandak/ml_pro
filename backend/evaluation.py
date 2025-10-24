from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score, mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except Exception:
    MLXTEND_AVAILABLE = False

def evaluate_supervised(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }

def evaluate_unsupervised(X, labels, model=None):
    metrics = {}
    if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette
        metrics['silhouette_score'] = silhouette_score(X, labels)
    if len(set(labels)) > 1:
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
    if model and hasattr(model, 'inertia_'):
        metrics['inertia'] = model.inertia_
    return metrics

def get_status_supervised(metrics):
    if metrics['accuracy'] > 0.8:
        return 'good'
    else:
        return 'bad'

def get_status_unsupervised(metrics):
    if 'silhouette_score' in metrics and metrics['silhouette_score'] > 0.1:
        return 'good'
    elif 'calinski_harabasz_score' in metrics and metrics['calinski_harabasz_score'] > 5:
        return 'good'
    elif 'davies_bouldin_score' in metrics and metrics['davies_bouldin_score'] < 3.0:
        return 'good'
    else:
        return 'bad'

def evaluate_regression(y_true, y_pred):
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred)
    }

def get_status_regression(metrics):
    if metrics['r2_score'] > 0.7:
        return 'good'
    else:
        return 'bad'


def cluster_elbow(X, k_range=range(2, 11), dataset_name=None):
    """
    Compute both inertia and silhouette scores for k in k_range using KMeans.
    Automatically handles common datasets like Mall_Customers.
    Returns: list of dicts with k, inertia, silhouette score and other metrics
    """
    # Handle Mall_Customers.csv specifically
    if isinstance(X, pd.DataFrame) and dataset_name == 'Mall_Customers.csv':
        possible_cols = ['Annual Income (k$)', 'Spending Score (1-100)', 
                         'Annual Income ($)', 'Spending_Score']
        existing_cols = [col for col in possible_cols if col in X.columns]
        if len(existing_cols) >= 2:
            X = X[existing_cols[:2]]
        else:
            raise ValueError("Mall_Customers.csv does not have expected columns.")

    # Convert to numpy array
    if isinstance(X, pd.DataFrame):
        X = X.select_dtypes(include=[np.number]).values  # numeric only

    results = []
    prev_inertia = None
    max_silhouette = -1
    best_k = None

    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = km.fit_predict(X)
        inertia = km.inertia_
        
        # Calculate improvement in inertia
        improvement = None
        if prev_inertia is not None:
            improvement = (prev_inertia - inertia) / prev_inertia
        prev_inertia = inertia
        
        # Calculate silhouette score for k >= 2
        silhouette = None
        if k >= 2:
            silhouette = silhouette_score(X, labels)
            if silhouette > max_silhouette:
                max_silhouette = silhouette
                best_k = k
        
        results.append({
            'k': k,
            'inertia': float(inertia),
            'improvement': float(improvement) if improvement is not None else None,
            'silhouette': float(silhouette) if silhouette is not None else None,
            'recommended': False  # Will be set after all calculations
        })

    # Find optimal k using both silhouette score and elbow method
    if best_k is not None:
        # Look for elbow point near the best silhouette score
        elbow_k = None
        for i in range(1, len(results)-1):
            current_improvement = results[i]['improvement']
            next_improvement = results[i+1]['improvement']
            # Significant drop in improvement (elbow)
            if current_improvement is not None and next_improvement is not None:
                if current_improvement > 0.2 and next_improvement < 0.1:
                    elbow_k = results[i]['k']
                    break
        
        # If elbow point and silhouette score suggest similar k (within ±1)
        if elbow_k is not None and abs(elbow_k - best_k) <= 1:
            # Choose the one with better silhouette score
            k_to_recommend = best_k
        else:
            # Prefer silhouette score as it's more reliable
            k_to_recommend = best_k
            
        # Mark the recommended k
        for result in results:
            if result['k'] == k_to_recommend:
                result['recommended'] = True
                break
                
    return results


def cluster_metrics(X, labels, model=None):
    """
    Compute silhouette, Calinski-Harabasz, Davies-Bouldin, and inertia metrics.
    """
    metrics = {}
    n_clusters = len(np.unique(labels))

    if 1 < n_clusters < len(X):
        try:
            metrics['silhouette_score'] = float(silhouette_score(X, labels))
        except:
            metrics['silhouette_score'] = None
        try:
            metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(X, labels))
        except:
            metrics['calinski_harabasz_score'] = None
        try:
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(X, labels))
        except:
            metrics['davies_bouldin_score'] = None
    else:
        metrics['silhouette_score'] = None
        metrics['calinski_harabasz_score'] = None
        metrics['davies_bouldin_score'] = None

    if model is not None and hasattr(model, 'inertia_'):
        metrics['inertia'] = float(model.inertia_)

    return metrics

def apriori_association(transactions, min_support=0.1, min_threshold=0.2):
    """Run apriori on a list of transactions (list of lists of items). Returns rules.
    
    Args:
        transactions: list of transactions, each a list of items
        min_support: minimum support threshold (0.1 = 10% of transactions)
        min_threshold: minimum confidence threshold (0.2 = 20% confidence)
        
    If mlxtend is not installed, returns a helpful error message.
    """
    if not MLXTEND_AVAILABLE:
        raise RuntimeError('mlxtend is not installed. Install it with `pip install mlxtend` to use association rules.')
    # Validate input transactions
    if not transactions or len(transactions) == 0:
        # Return empty DataFrames to indicate no frequent itemsets/rules
        return pd.DataFrame(), pd.DataFrame()

    # Debug print to see what's coming in
    print(f"Initial transactions head (first 5): {transactions[:5]}")
    print(f"Number of transactions: {len(transactions)}")

    # Clean transactions: ensure list-of-lists of non-empty string items
    cleaned = []
    for t in transactions:
        if t is None:
            continue
        # If transaction is a single string representing comma-separated items, split it
        if isinstance(t, str):
            items = [it.strip() for it in t.split(',') if it.strip()]
        else:
            # try to iterate; convert items to strings and strip
            try:
                items = [str(it).strip() for it in t if it is not None and str(it).strip()]
            except Exception as e:
                print(f"Error processing transaction {t}: {str(e)}")
                # skip invalid transaction entries
                items = []
        if items:
            cleaned.append(items)
            
    # Debug print after cleaning
    print(f"Cleaned transactions head: {cleaned[:5]}")
    print(f"Number of cleaned transactions: {len(cleaned)}")

    if not cleaned:
        return pd.DataFrame(), pd.DataFrame()

    try:
        te = TransactionEncoder()
        te_ary = te.fit_transform(cleaned)
        df = pd.DataFrame(te_ary, columns=te.columns_)
    except Exception as e:
        # If transaction encoder fails, return empty DataFrames
        return pd.DataFrame(), pd.DataFrame()

    # If DataFrame has no columns or no positive entries, no frequent itemsets
    if df.shape[1] == 0 or df.values.sum() == 0:
        return pd.DataFrame(), pd.DataFrame()

    # More debug prints
    print(f"DataFrame shape before apriori: {df.shape}")
    print(f"Sample columns: {df.columns[:5]}")
    print(f"Sample values:\n{df.head()}")
    
    freq = apriori(df, min_support=min_support, use_colnames=True)
    print(f"Frequent itemsets found: {len(freq) if freq is not None else 0}")
    
    if freq is None or freq.empty:
        print("No frequent itemsets found!")
        return freq if freq is not None else pd.DataFrame(), pd.DataFrame()

    rules = association_rules(freq, metric='confidence', min_threshold=min_threshold)
    print(f"Number of rules generated: {len(rules)}")
    if not rules.empty:
        print("Sample rules:\n", rules.head())
    return freq, rules
