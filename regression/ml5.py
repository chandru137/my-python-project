'''
What is Boosting?
Boosting is an ensemble machine learning technique that combines multiple weak learners to create a strong learner.

--types 
1)AdaBoost:Focues on incorrectly predicted samples
2)gradient boosting:Corrects errors using gradients and loss functions
3)XGBoost:optimized gradient boosting with regularization
4)LightGBM:fast boosting for large datasets 
5)CatBoost:boosting designed to handle categorical features effectively
'''
#====================================================
#               Ada boosting
#====================================================
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
X=[[1],[2],[3],[4],[5]]
Y=[20,30,40,50,60]
tree=DecisionTreeRegressor(
    max_depth=3                  #creates a simple decision tree
)
model=AdaBoostRegressor(
    estimator=tree,               #base regression moddel
    n_estimators=100,             #number of weak learners
    learning_rate=0.1,            #controls contribution of each lerner
    loss='linear',                #we can also use 'square' or 'exponential'
    random_state=42               #reproducible results
)
model.fit(X,Y)
prediction=model.predict([[6]])
print(prediction[0])
# ====================================================
#                Gradient boosting
# ====================================================
from sklearn.ensemble import GradientBoostingRegressor
# Input: Study hours
X = [[1], [2], [3], [4], [5]]
# Output: Actual marks
y = [20, 30, 40, 50, 60]
# Create model
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=1,
    random_state=42
)
# Train model
model.fit(X, y)
# Predict for 6 study hours
prediction = model.predict([[6]])
print("Predicted Marks:", prediction[0])

# ====================================================
#                XGBoost
#====================================================
from xgboost import XGBRegressor
# Input: Study hours
X = [[1], [2], [3], [4], [5]]
# Output: Student marks
y = [20, 30, 40, 50, 60]
# Create XGBoost model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=2,
    objective="reg:squarederror",            #uses square error for regression tasks
    random_state=42
)
# Train the model
model.fit(X, y)
# Predict marks for 6 study hours
prediction = model.predict([[6]])
print("Predicted Marks:", prediction[0])


# ====================================================
#                LightGBM
# ====================================================
from lightgbm import LGBMRegressor

# Input: Study hours
X = [[1],[2],[3],[4],[5]]

# Output: Student marks
y = [20, 30, 40, 50, 60]

# Create model
model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    num_leaves=7,
    min_child_samples=2,
    random_state=42,
    verbosity=-1
)
# Train model
model.fit(X, y)

# Predict marks for 6 study hours
prediction = model.predict([[6]])

print("Predicted Marks:", prediction[0])

#=================================================
#       CatBoost
#=================================================
from catboost import CatBoostRegressor
# Input data
X = [
    [1, "CSE"],
    [2, "AIML"],
    [3, "CSE"],
    [4, "AIML"],
    [5, "CSE"]
]

# Actual marks
y = [20, 30, 40, 50, 60]

# Create model
model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=3,
    loss_function="RMSE",
    verbose=0
)

# Train model
model.fit(X, y, cat_features=[1])

# Prediction for 6 hours, AIML
prediction = model.predict([[4, "CSE"]])
print("Predicted Marks:", prediction[0])