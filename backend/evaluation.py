from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score, mean_squared_error, mean_absolute_error, r2_score

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
