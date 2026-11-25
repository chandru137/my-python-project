'''decision tree regression:algorithm used to predict a continuous numerical value like house price salary temprature sales stock price and student marks
how it's work?
-starts with all training data.
-tries different splitting points
-chooses a split that reduce prediction error.
-repeats the process on each group.
-stops when the stopping conditions are reached.
-uses the average target value in a leaf as the prediction.

linear regression                    vs       Decision tree regression
-fits the straight line               | -creats rule-based regions
-assumes a linear relationship        | -can model nonlinear relationships
-formula-based                        | -tree/split-based
-sensitive to outliers                | -generally less sensitive
-can extrapolate                      | -usually poor at extrapolation

A decision tree does not necessarily predict exactly one training value.
for example:
            marks:65,72,80
            prediction:(65+72+80)/3=72.33
'''

from sklearn.tree import DecisionTreeRegressor
A=[[100],[102],[105],[107],[110],[108],[112],[115]]             #previous day stock price
B=[102,105,107,110,108,112,115,118]                             #today's stock price
model=DecisionTreeRegressor()          
model.fit(A,B)
c=float(input("Enter The Previous Day Stock Price:"))
prediction=model.predict([[c]])
print("predited stock price",prediction[0])

'''
NOTE:background flow:Training data-->Previous day stock price-->Today's stock price-->Model Fit-->Decision tree regression model-->Predicted stock price
arrange the data A in incressing order 
A    |B    |A(n+1)-A(n) |
     |     |=X          |
_____|_____|____________|             n=1,2,,,,,,7 
100  |102  |            |     ex:102-100=101
     |     |101         |
102  |105  |            |
     |     |103.5       |
105  |107  |            |
     |     |106         |
107  |110  |            |
     |     |107.5       |
108  |112  |            |
     |     |109         |
110  |108  |            |
     |     |111         |
112  |115  |            |
     |     |113.5       |
115  |118  |            |
-------------------------
slipt the data into two parts by the user input if c=111 by X<=111 and X>111
first check the condition
101<111,103.5<111,106<111,107.5<111,109<111? yes
111=111? Yes
113.5<111? no
group1:102,105,107,110,112,108
group2:115,118

group1: b1=(102+105+107+110+112+108)/6=107.333(the predected value is approximately 108)
group2: b2=115+118/2=116.5
'''


'''
Random Forest Regression:
                         that uses multiple decision trees and combines their predictions to preduce a final predection.
                         
advantages of Random Forest Regression:
1. Accuracy: Random Forest Regression often provides higher accuracy compared to a single decision tree by reducing overfitting and capturing complex relationships in the data.
2. Robustness: It is less sensitive to noise and outliers in the data, making it more robust for real-world applications.
3. Versatility: Random Forest Regression can handle both numerical and categorical features, making it suitable for a wide range of regression tasks.
4. Feature Importance: It can provide insights into the importance of different features in predicting the target variable, which can be valuable for feature selection and understanding the underlying data.
5. Parallelization: Random Forest can be easily parallelized, allowing for faster training on large datasets.

important: Random Forest Regression does not always produce more accurate results than other regression models.always evaluate mae mse rmse and r2.
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# Training data
# X = Study hours
X = [[1], [2], [3], [4], [5], [6], [7], [8]]

# y = Marks
y = [20, 25, 35, 45, 50, 60, 70, 85]

# Create the model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X, y)
y_pred=model.predict(X)            #predication for traning the data->for metrics
print("predication:",y_pred)             
print("MAE:",mean_absolute_error(y,y_pred))   #finding the of MAE,MSE,R2
print("MSE:",mean_squared_error(y,y_pred))    #MAE and MSE  e- show error is not 
print("R2:",r2_score(y,y_pred))              #R2=1 perfect fit


# Predict marks for 5.5 study hours
prediction = model.predict([[5.5]])

print("Study hours:", 5.5)
print("Predicted marks:", round(prediction[0], 2))


'''
NOTE:it's not give accurate result because the data is not enough for training the model.'''

'''
How Does Random Forest Make a Prediction? Imagine three decision trees predict marks for a student:
Tree 1 predicts 50.
Tree 2 predicts 55.
Tree 3 predicts 60.
The average prediction is:prediction= (50+55+60)/ 3
Prediction = 55

 
5. Mathematical Formula for Random Forest Regression

The final prediction is the average of the predictions made by all the trees:
                 N
    y' = (1/N) * Σ T; (X)
                 i=1
Where:

y'= Final predicted marks.

• N = Number of decision trees.

• T; (X) = Prediction made by the i-th tree.

X = Input study hours, which is 5.5 in your example.

'''