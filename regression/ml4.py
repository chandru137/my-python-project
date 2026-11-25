'''
KNN Regression:
               the model predicts by looking at the values of the nearest data point
                
KNN regression follows these steps:
1-Store the training data.
2-Calculate the distance between the new input and existing data points.
3-Find the K nearest neighbors.
4-Calculate the average of their target values.
5-Return that average as the prediction.
'''
from sklearn.neighbors import KNeighborsRegressor

# Training data: study hours
X = [[1], [2], [3], [4], [5]]

# Target data: student marks
y = [20, 30, 40, 50, 60]

# Create KNN Regression model
model = KNeighborsRegressor(n_neighbors=2)

# Train the model
model.fit(X, y)

# Predict marks for 3.5 study hours
prediction = model.predict([[3.5]])

print("Predicted marks:", prediction[0])

'''
formula for KNN regression:y' = (1/K) * Σ y_i
y'=(sum of neighbors' marks)/k
Where:
- y' is the predicted value
- K is the number of nearest neighbors
- yi is the target value of the i-th nearest neighbor

A smaller k can be sensitive to individual data points, while a larger k can smooth out predictions but may overlook local patterns.
The choice of k is crucial for the performance of the KNN regression model.
'''





'''
SVR(support vector Regression):it's used to predict continue numerical values.

How does SVR work?
SVR tries to find a function that predicts values while allowing a certain amount of error, called the epsilon (ε) margin.
Unlike ordinary Linear Regression, SVR does not try to minimize every prediction error.
It focuses on errors outside the epsilon margin and uses support vectors to determine the model.
Imagine the actual data points are scattered around a line.
The middle line represents the prediction function.
The upper and lower boundaries represent the epsilon margin.
Points outside the margin can influence the model's optimization.
Points on or outside the margin may become support vectors.

Types of SVR
SVR can use different kernels to learn different relationships.
Kernel                     |Purpose                                   |Example use
Linear                     |Models a linear relationship              |Approximately linear price trends
Polynomial                 |Models polynomial relationships           |Curved relationships between features and prices
RBF                        |Learns flexible nonlinear relationships   |Complex relationships in house-price data
Sigmoid                    |Models sigmoidal relationships            |Specialized applications where this kernel is appropriate

7. When should you use SVR?
Use SVR when:
You want to predict a continuous numerical value.
Your dataset has a potentially complex relationship between inputs and outputs.
You want to compare a margin-based regression method with Linear Regression, KNN, or Random Forest.
You have time to tune parameters such as C, epsilon, and gamma.

Advantages and Disadvantages of SVR
Advantages
Can model linear and nonlinear relationships.
Uses an epsilon margin to control the errors it penalizes.
Can work well on small and medium-sized datasets.
Can generalize well when its parameters are tuned appropriately.
Disadvantages
Feature scaling is usually important.
Choosing the right kernel and parameters can be difficult.
Training can become computationally expensive on large datasets.
Predictions are less interpretable than a simple linear regression equation.

'''
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Training data
X = [[1], [2], [3], [4], [5]]
y = [20, 30, 40, 50, 60]

# Create SVR model
model = make_pipeline(
    StandardScaler(),
    SVR(kernel="linear", C=100, epsilon=0.1)
)

# Train the model
model.fit(X, y)

# Predict marks for 3.5 study hours
prediction = model.predict([[3.5]])

print("Predicted marks:", prediction[0])
'''
A.Linear SVR:
            SVR(kernel="linear")
            Use it when the relationship is approximately linear.
B. Polynomial SVR:
            SVR(kernel="poly", degree=3)
            Use it when a polynomial-shaped relationship may be suitable.
            The degree=3 parameter sets the polynomial kernel's degree.
C. RBF SVR:
           SVR(kernel="rbf")
           RBF stands for Radial Basis Function.
           It is a common choice when the relationship is nonlinear and you do not know the exact shape in advance.

NOTE:SVR Mathematical Example
Suppose you have the following training data:
Study hours (X)    | Actual marks (y)
1                  | 20
2                  | 30
3                  | 40
4                  | 50
5                  | 60
You want to predict the marks for a student who studies for 3.5 hours.
For a linear SVR model, the prediction function is:
Where:
w = coefficient or slope
x= input value
b = intercept
You want to predict the marks for a student who studies for 3.5 hours.
For a linear SVR model, the prediction function is:
Where:
w = coefficient or slope
x= input value
b = intercept
Suppose a trained SVR model learns the function:
For x=3.5:
Y=wx+b
y=10(3.5)+10
Y=35+10
Y=45
'''