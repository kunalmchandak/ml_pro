from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline
try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    XGBRegressor = None
    XGBClassifier = None
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except ImportError:
    LGBMRegressor = None
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

SUPERVISED_ALGORITHMS = {
    # Classification algorithms
    'logistic_regression': LogisticRegression,
    'random_forest': RandomForestClassifier,
    'svm': SVC,
    'knn': KNeighborsClassifier,
    'decision_tree': DecisionTreeClassifier,
    'gaussian_nb': GaussianNB,
    'multinomial_nb': MultinomialNB,
    'bernoulli_nb': BernoulliNB,
    'gradient_boosting_classifier': GradientBoostingClassifier,
    'xgboost_classifier': XGBClassifier if 'XGBClassifier' in locals() else None,
    'lightgbm_classifier': LGBMClassifier if 'LGBMClassifier' in locals() else None,
    # In supervised.py, change the condition
    'catboost_classifier': CatBoostClassifier if CatBoostClassifier is not None else None,
    
    # Regression algorithms
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
