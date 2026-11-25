#for 1 coefficient errors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,root_mean_squared_error,mean_absolute_error,mean_squared_error
x=[[1],[2],[3],[4],[5]]
y=[10,20,30,40,50]
model=LinearRegression()
model.fit(x,y)
zibra_1=model.predict(x)
print(r2_score(y,zibra_1))                #1 output no error of r2
print(root_mean_squared_error(y,zibra_1)) #rmse,mae,msr e- output data model correct
print(mean_absolute_error(y,zibra_1))
print(mean_squared_error(y,zibra_1))
'''Error detected   |Possible reason                              |What you should do
____________________|_____________________________________________|____________________________________________________
High MAE            |Predictions are far from actual values       |Check data quality and features
High MSE            |Some predictions have large errors           |Investigate large residuals
High RMSE           |Large prediction errors                      |Inspect outliers and improve the model
Low R²              |Model does not explain the data well         |Check features and try another model
Overfitting         |Model memorizes training data                |Simplify the model or use regularization
Underfitting        |Model is too simple                          |Try a more suitable model
Outliers            |Unusual observations                         |Verify the data before deciding what to do'''
# for coefficient 2 or more 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
a=[[1,2],[2,3],[3,4],[4,5],[5,6]]
b=[10,20,30,40,50]
model=LinearRegression()
model.fit(a,b)
zibra_2=model.predict(a)
print(r2_score(b,zibra_2))
print(root_mean_squared_error(b,zibra_2))
print(mean_absolute_error(b,zibra_2))
print(mean_squared_error(b,zibra_2))