#finding the model in overfitting or underfitting
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Training data
X = [[1], [2], [3], [4], [5],
     [6], [7], [8], [9], [10]]
y = [10, 20, 30, 40, 50,
     60, 70, 80, 90, 100]
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Calculate errors
train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, test_pred)

# Calculate R2
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

# Display results
print("Training MSE:", train_mse)
print("Testing MSE:", test_mse)

print("Training R2:", train_r2)
print("Testing R2:", test_r2)

# Identify model behavior
if train_mse > 100 and test_mse > 100:
    print("Underfitting")

elif test_mse > train_mse * 3:
    print("Possible Overfitting")

else:
    print("Model may be fitting well")