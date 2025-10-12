# TODO: Add Regression Algorithms and Datasets

## Backend Changes
- [x] Update backend/algorithms/supervised.py: Import regression classes (LinearRegression, Ridge, Lasso, PolynomialFeatures + LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, XGBRegressor, LGBMRegressor, SVR, KNeighborsRegressor), add to SUPERVISED_ALGORITHMS dict.
- [x] Update backend/datasets.py: Import fetch_california_housing, fetch_openml; add new regression datasets (California Housing, Boston, Concrete Compressive Strength, Bike Sharing, Energy Efficiency, Student Performance, Air Quality, Weather History, CO2 Emission, Life Expectancy) to DEFAULT_DATASETS with 'types': ['regression'], descriptions, features, samples.
- [x] Update backend/evaluation.py: Add evaluate_regression function with mean_squared_error, mean_absolute_error, r2_score; add get_status_regression function.
- [x] Update backend/main.py: Add regression algorithms to ALGORITHMS['supervised'] with 'task': 'regression', 'compatible_types': ['regression']; update train_model to check 'task' and use evaluate_regression for regression.

## Frontend Changes
- [x] Update frontend/src/components/Dashboard.js: Add regression algorithm options to the supervised optgroup in the select element.
- [x] Update frontend/src/components/Visualization.js: Update detection logic to check for 'r2_score' for regression; add regression metrics to chart and details display.

## Followup
- [ ] Install additional packages (xgboost, lightgbm, kaggle) via pip install -r backend/requirements.txt
- [ ] Set up Kaggle API key for dataset downloads (run kaggle config set -n username -v YOUR_USERNAME and kaggle config set -n key -v YOUR_API_KEY)
- [ ] Test training a regression model (e.g., linear_regression on boston_housing), verify metrics in visualization.
- [ ] Check for errors, ensure frontend renders correctly.
