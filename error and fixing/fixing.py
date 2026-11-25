'''the model automatically selects the best model from a set of regression models, tunes its hyperparameters, 
and evaluates its performance on a test dataset. It also reports residuals and saves the tuned model for future use.


input data-->data validation-->train/test split-->model selection-->hyperparameter tuning-->model evaluation-->residual analysis-->save model'''
import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------
# 1. Create sample regression dataset
# ------------------------------------
X= np.array([
    [1,10,20,30,40],
    [2,12,22,32,42],
    [3,13,23,33,43],
    [4,14,24,34,44],
    [5,15,25,35,45]
])
y= np.array([35, 42, 50, 61, 70])

X = pd.DataFrame(X, columns=[                                          #the feature are columns of the dataset if x is [1,2,3,4,5] 
    "Feature1",                                                         #then the column name is Feature1
    "Feature2",                                                          #than x=pd.DataFrame(X, columns=["Feature1"])
    "Feature3",
    "Feature4",
    "Feature5"
])

# ------------------------------------
# 2. Split the dataset
# ------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# ------------------------------------
# 3. Define regression models
# ------------------------------------

models = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression())
    ]),

    "Polynomial Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(
            degree=2,
            include_bias=False
        )),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Ridge Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Lasso Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1, max_iter=10000))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeRegressor(
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# ------------------------------------
# 4. Evaluate models using cross-validation
# ------------------------------------

cv = KFold(
    n_splits=min(5, len(X_train)),
    shuffle=True,
    random_state=42
)

results = []

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    cv_mae = -scores.mean()

    results.append({
        "Model": name,
        "CV_MAE": cv_mae
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("CV_MAE")

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# ------------------------------------
# 5. Select the best model
# ------------------------------------

best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]

print("\nSelected model:", best_name)

# ------------------------------------
# 6. Tune hyperparameters
# ------------------------------------

parameter_grids = {
    "Linear Regression": {},

    "Polynomial Regression": {
        "poly__degree": [1, 2, 3],
        "model__alpha": [0.01, 0.1, 1.0, 10.0]
    },

    "Ridge Regression": {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
    },

    "Lasso Regression": {
        "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
    },

    "Decision Tree": {
        "model__max_depth": [None, 3, 5, 10],
        "model__min_samples_leaf": [1, 2, 5]
    },

    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 2, 5]
    }
}

grid = GridSearchCV(
    estimator=best_model,
    param_grid=parameter_grids[best_name],
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1
)

grid.fit(X_train, y_train)

improved_model = grid.best_estimator_

print("\nBest parameters:", grid.best_params_)
print("Best cross-validation MAE:", -grid.best_score_)

# ------------------------------------
# 7. Evaluate the tuned model on test data
# ------------------------------------

predictions = improved_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
if len(y_test) >= 2:
    r2 = r2_score(y_test, predictions)
else:
    r2 = None

print("\nFINAL TEST RESULTS")
print("-------------------------")
print("MAE :", round(mae, 3))
print("MSE :", round(mse, 3))
print("RMSE:", round(rmse, 3))
if r2 is not None:
    print("R2  :", round(r2, 3))
else:
    print("R2  :", "Not Available")

# ------------------------------------
# 8. Report residuals
# ------------------------------------

residuals = np.array(y_test) - np.array(predictions)

print("\nERROR ANALYSIS")
print("-------------------------")
print("Mean residual:", round(residuals.mean(), 3))
print("Maximum absolute error:",
      round(np.abs(residuals).max(), 3))

# ------------------------------------
# 9. Save the tuned model
# ------------------------------------
# import joblib

# joblib.dump(improved_model, "best_regression_model.pkl")
# print("\nTuned model saved as 'best_regression_model.pkl'")
