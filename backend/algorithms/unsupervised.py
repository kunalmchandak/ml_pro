from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch

# Unsupervised algorithm constructors mapping
UNSUPERVISED_ALGORITHMS = {
    'kmeans': KMeans,
    'dbscan': DBSCAN,
    'agglomerative': AgglomerativeClustering,
    'birch': Birch,
}

# Association-type algorithms (require transaction-style data). These are handled
# differently (not sklearn estimators). We keep a small mapping for callers to
# detect available association methods; actual implementations live in
# backend.evaluation (apriori/association rules) and may require extra packages.
ASSOCIATION_ALGORITHMS = {
    'apriori': 'apriori',
    'fp_growth': 'fp_growth'
}
