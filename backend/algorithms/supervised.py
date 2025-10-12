from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

SUPERVISED_ALGORITHMS = {
    'logistic_regression': LogisticRegression,
    'random_forest': RandomForestClassifier,
    'svm': SVC,
    'knn': KNeighborsClassifier,
    'decision_tree': DecisionTreeClassifier,
    'linear_regression': LinearRegression,
    'ridge': Ridge,
    'lasso': Lasso,
    'decision_tree_regressor': DecisionTreeRegressor,
    'random_forest_regressor': RandomForestRegressor,
    'gradient_boosting': GradientBoostingRegressor,
    'xgboost': XGBRegressor if XGBRegressor else None,
    'lightgbm': LGBMRegressor if LGBMRegressor else None,
    'svr': SVR,
    'knn_regressor': KNeighborsRegressor,
}
