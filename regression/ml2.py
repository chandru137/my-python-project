'''Ridge Regression:Ridge Regression is a type of linear regression that adds a penalty to large model coefficients.
it's useful when:
1-many input features. 2-multicollinearity(the features are highly related with each other). 3-when the model is overfitting.
linear Regression-->fit the data-->Ridge regression-->fit the data-->control large coefficients

what is a penalty?
A penalty is an extra mathematical cost added to the regression error when the model has large co-efficients.

using of alpha:
               small alpha-->weak penalty-->more complex model-->overfitting
               large alpha-->strongh penalty-->simpe model-->underfitting-->co-efficient shrink

Ridge Regression                               vs                   linear regression
-adds coefficient penalty                       |                    -no coefficient penalty
-helps to reduce overfitting                    |                    -overfitting is possible
-coefficient values are small                   |                    -coefficient values can be large
-regluarization parameter is used for the model |                    -simple model

ridge regression formula:y=b0+b1x1+b2x2+......+bnxn+λ(b1^2+b2^2+......+bn^2)
where λ is the regularization parameter that controls the amount of shrinkage applied to the coefficients.'''

#this program automaticaly apply best alpha value by the model_selection
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
X=[[2,60,45],                                             #study hours,attendance,assignment marks
   [3,70,50],                                  
   [4,75,60],
   [5,85,70],
   [6,90,80]
   ]
Y=[50,58,68,78,88]                                 #final marks
params={
    "alpha":[0.001,0.01,0.1,1,10,100,100]          #alpha value to test
}
grid=GridSearchCV(                                 #automaticaly find the best alpha
    Ridge(),
    params,
    cv=5,                                          #datasets have 5 students
    scoring="neg_mean_squared_error"               #lowest average cross validation mse among the condition
)
grid.fit(X,Y)
model=grid.best_estimator_
p=int(input("Enter the study hours:"))
q=int(input("Enter the attendance in percentage:"))
r=int(input("Enter the assignment marks:"))
if(p<=24 and q<=100 and r<=100):
 print("Best Alpha:",grid.best_params_["alpha"])
 print("Best MSE:",-grid.best_score_)
 print("Coefficients:",model.coef_)
 print("Intercept:",model.intercept_)
 prediction=model.predict([[p,q,r]])
 print("Predication marks",prediction[0])
else:
 print("Invailed Input:")

'''
study hours(x1)     attendance(x2)     assignment marks(x3)    final marks(y1)
2                   60                 45                      50
3                   70                 50                      58
4                   75                 60                      68
5                   85                 70                      78
6                   90                 80                      88
alpha=1 means the regularization parameter that controls the amount of shrinkage applied to the coefficients.
1.ridge equation:
our model equation: y*=bo+b11x1+b2x2+b3x3
   ridge equation: Y=bo+b11x1+b2x2+b3x3+λ(b1^2+b2^2+b3^2)

2.calculate the means
x1=(2+3+4+5+6)/5=4
x2=(60+70+75+85+90)/5=76
x3=(45+50+60+70+80)/5=61
y=(50+58+68+78+88)/5=68.4

now calculate the centered values:
formula: x1-x1mean,x2-x2mean,x3-x3mean,y-ymean  or
         x1-4     ,x2-76     ,x3-61     ,y-68.4
x1-x1mean     x2-x2mean     x3-x3mean     y-ymean
-2            -16            -16           -18.4
-1            -6             -11           -10.4
0             -1              -1            -0.4
1              9               9             9.6
2             14              19            19.6

the centered matrix are:
Xc=|-2  -16  -16|                 Yc=|-18.4|
   |-1  -6   -11|                    |-10.4|
   |0   -1   -1 |                    | -0.4|
   |1    9    9 |                    |  9.6|
   |2    14   19|                    | 19.6|

Formula for ridge regression is:
b=(Xc^T*Xc+λI)^-1*Xc^TYc
λ=alpha=1

3.calculate 
transpose of Xc:
Xc^T=|-2   -1  0   1   2|        Xc^T*Yc=|-2  -1   0   1   2||-18.4|           Xc^T*Yc=|96 |
     |-16  -6 -1   9  14|                |-16 -6  -1   9  14||-10.4|                   |728|
     |-16 -11 -1   9  19|                |-16 -11 -1   9  19|| -0.4|                   |868|
                                                             |9.6  |
                                                             |19.6 |

know calculate Xc^T*Xc:                                          
Xc^T*Xc=|10  90  9   10|
        |90  354  9  90|   
        |9   90  9   10|
Xc^T*Xc+λI=|10  90  9   10|+|1 0 0|=  |11  90   9  10|
           |90  354  9  90| |0 1 0|   |90  355  9  90|
           |9   90  9   10| |0 0 1|   |9   90  10  10|

Xc^T*Xc+λI=|11  90   9  10||b1|=|96 |
           |90  355  9  90||b2| |728|
           |9   90  10  10||b3| |868|

equation1: 11b1 + 90b2 + 9b3 = 96
equation2: 90b1 + 355b2 + 9b3 = 728
equation3: 9b1 + 90b2 + 10b3 = 868

solve the equations by elimination method:b1=0.221909,b2=0.383845,b3=0.719673
4.calculate the intercept:
b0=y1-b1x1-b2x2-b3x3
b0=-5.55995

the final equation becames:Y*=-5.5595+0.221909x1+0.383845x2+0.719673x3 put value of study hours(x1),attendance(x2),assignment marks(x3)
'''


'''Lasso Regression:(Least absolute shrinkage and selection operator)is a linear regression technique that adds a penalty to the size
of the model coefficients.
~it is useful when you have many features and want the model to automatically select the important ones 

Use of lasso
1.remove unnecessary features.    2.handle dataset with features   3.reduce overfitng 4.Make the model easier to interpret'''
from sklearn.linear_model import Lasso
A=[
    [1.0,2,20],
    [1.5,3,15],
    [2.0,3,10],
    [2.5,4,5],
    [3.0,4,3],
    [3.5,5,2]
]
B=[50,70,85,110,135,160]
model =Lasso(alpha=0.1)
model.fit(A,B)#type:ignore
print(model.intercept_)
print(model.coef_)
new_house=([[2.8,4,4]])
prediction=model.predict(new_house)#type:ignore
print("Predection=",prediction)
'''By Maths to solve the problem 
x1  |x2  |x3  |y  |
1.0 |2   |20  |50 |
1.5 |3   |15  |70 |
2.0 |3   |10  |85 |
2.5 |4   |5   |110|
3.0 |4   |3   |135|
3.5 |5   |2   |160|                       alpha=1

for linear equation formula
y=b0+b1x1+b2x2+b3x3

lasso penalty formula:
Xc*Xc^TB=Xc^T*yc-n(alpha)|1|--(1)
                         |1|
                         |1|
x1=[(1.0+1.5+2.0+2.5+3.0+3.5)/6]=2.25
x2=[(2+3+3+4+4+5)/6]=3.5
x3=[(20+15+10+5+3+2)/6]=9.1667
y=[(50+70+85+110+135+160/6)]=101.6667

calculate the center the data
Xc=|-1.25  -1.5  10.8333|
   |-0.75  -0.5  5.8333 |
   |-0.25  -0.5  0.8333 |
   |0.25    0.5  -4.1667|
   |0.75    0.5  -6.1667|
   |1.25    1.5  -7.1667|

yc=|-51.667|
   |-31.667|
   |-16.667|
   |8.33333|
   |33.3333|
   |58.3333|

n=6 (numbers of element)  alpha=0.1

apply and calculate from equation--(1)
|4.375   4.75   -32.75||b1|=|192.5   |-(0.1*6)|1|
|4.75    5.5    -35.5 ||b2| |210     |        |1|
|-32.75  -35.5  258.83||b3| |-1416.66|        |1|

b1=51.2293      b2=2.8341     b3=1.3951

find the intercept:b0=y-b1x1-b2x2-b3x3   y=101.6667
the b0=-36.3073

the eqution becames:
y=-36.3073+51.2293x1+2.8341x2+1.3951x3'''

#NOTE:YOU CAN TAKE FROM THE USERS INPUT BY INPUT KEY AND FLOAT 


'''
ELASTIC NET REGRESSION:Elastic Net regression is a linear regression technique that combines both L1 (Lasso) 
                       and L2 (Ridge) regularization penalties to shrink coefficients and perform feature selection
                       '''

from sklearn.linear_model import ElasticNet

# 1. Create a tiny dataset (3 samples, 2 features)
# Feature 1 and Feature 2 are highly correlated
X = [[1, 1], [2, 2], [3, 3]]
y = [2, 4, 6]

# 2. Define the model with fixed hyperparameters
# alpha = total penalty strength, l1_ratio = 50% Lasso / 50% Ridge
model = ElasticNet(alpha=0.1, l1_ratio=0.5)

# 3. Train the model
model.fit(X, y)

# 4. Predict the output for a new data point [4, 4]
prediction = model.predict([[4, 4]])

print(f"Prediction for: {prediction[0]:.2f}")
print(f"Weights/Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.2f}")
'''
The dataset is:

Sample

Feature 1 (1)  |Feature 2 (2) | Target

1               |1              |2
2               |2              |4
3               |3              |6  

New input
4
4
?

Parameters:
a = 0.1,
l1_ratio = 0.5


2. Step 1: Understand the regression equation
For two features, the regression equation is:y'=b0+b1x1+b2x2

Where:
bo = intercept
b1 = coefficient of Feature 1
b2 = coefficient of Feature 2
X1, X2 = input features
y' = predicted output
We need to find bo, b1, b2.

3. Step 2: Find the relationship between the features and target

Our observations are:

1+1=2

2+2=4

3+3=6

Therefore, the target follows:
Y=x1+x2
Without regularization, one possible solution is:b1=1,b2=1, b0=0
However, ElasticNet applies regularization, so its fitted coefficients will generally differ from these values.

Also, the two features are perfectly correlated:
X1 = X2 in every training sample. Consequently, the
data alone cannot determine how to divide the
coefficient between the two features.


4. Step 3: Understand the ElasticNet mathematical formula
ElasticNet minimizes the following objective:
          n                    p                       p
j=(1/2n) * Σ(y - y'i)^2 + αl * Σ|bj| + [ α(1 - l)/2] * Σ|bj^2|
          i=1                  j=1                     j=1
Where:
n = 3: number of samples
p = 2: number of features
alpha = 0.1: regularization strength
l=0.5: L1 ratio

The first term measures prediction error.
The second term is the Lasso penalty.
The third term is the Ridge penalty.

Substitute the parameter values:

al = 0.1(0.5) = 0.05
a(1-l)/2=0.1(1-0.5)/2= 0.025
                                      3
Thus, the objective becomes:j=(1/6) * Σ(y - y'i)^2 +0.05(|b1|+|b2|) + 0.025(b1^2+b2^2)
                                      i=1
Notice that the intercept is not penalized.
Finding the coefficients

Because the two features are identical, the optimal coefficients are equal:b1=b2=c
The prediction equation becomes:y'=bo+2cx

The target values are exactly twice the feature values.
Therefore, the unregularized intercept is zero, and the residuals are zero when bo = 0 and c = 1.

But ElasticNet penalizes large coefficients. Minimizing the objective gives:

C=1/1 +0.1(1- 0.5) +0.1(0.5)

Simplifying:

C=1/1+ 0.05+ 0.05
c≈ 0.90909

Therefore, the theoretical coefficients are:

And the intercept is:
b₁ ≈ 0.90909
b2 ≈ 0.90909
b0=0
And the intercept is:

These results assume the standard scikit-learn
ElasticNet objective, which does not standardize the
features automatically.

5. Step 4: Calculate the prediction
Your new input is:
x1 = 4, x2 =4

Use the regression equation:
        y'=b0+b1x1+b2x2
Substitute the coefficients:
y'= 0+ (0.90909)(4) + (0.90909)(4)

Calculate each term:0.90909*4=3.63636
y'=3.63636+3.63636
y'=7.27273
'''