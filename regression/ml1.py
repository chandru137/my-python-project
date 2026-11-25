'''what is machine learning🤔?
-machine learning allows a computer to learn patterns from data and make prediction without explicitly programming every rule'''


'''traditional programming:Data+rules-->output
machine learning:Data+Answer-->ML model
                       |
                       prediction'''


'''💻types of machine learning
1.supervised learning sub types a.regression b.classification
2.unsupervised learning sub types a.clustering b.dimensionality reduction
3.reinforcement learning'''

#DataSet:collection of information used by ml system

#1.supervised learning:in supervised learning ,the training data contains the correct answer
'''======================================================================================================================================
a--regression:it predicts a numerical value like(sales,house price,salary,stock) sub types a.simple linear regression b.multiple 
linear regression c.ploynomial regression
========================================================================================================================================'''


#first algorithm:linear regression 💫(y=mx+c) example for simple liner regression 
# first we have to install the sklearn library by using pip install scikit-learn

from sklearn.linear_model import LinearRegression                       #(mini project)
import matplotlib.pyplot as plt 
X=[[1],[2],[3],[4],[5]]        #student study hours             
Y=[35,42,50,61,70]              #obtained marks                 
model = LinearRegression()
model.fit(X,Y)
q=float(input("Enter Your Study hours Per Day:"))                        #user input
if q<=24:
 x_pred=model.predict([[q]])       #type:ignore (ignore the type)        #user predection
 x_line=model.predict([[1],[q]])     # type: ignore                     #line in the graph 
 plt.scatter([1,2,3,4,5],Y,color='blue',marker='H',label='Data')         #points of data

 plt.plot([1,q],x_line,color='red',label='line')                        #line of the graph 

 plt.scatter([q],x_pred,color='green',marker='*',label='Predection')    #predected data

 plt.xlabel('Stydent marks')
 plt.ylabel('Predected Marks')
 plt.title('Marks predection')
 plt.legend()
 plt.show()
 print("Predection marks:",x_pred[0])
else:
 print("Invailed study hours:")

'''predect by using maths
step1:calculate the mean of X and Y X=[(1+2+3+4+5)/5]=3=x+    #Y=[(35+42+50+61+70)/5]=51.6=y+

step2:calculate the slope(m)= semasen of (X-x+)(Y-y+)
                               -----------------------
                               semasen of (X-x+)^2
X|Y |X-x+|Y-y+ |(X-x+)*(Y-y+)|(X-x+)^2 |
1|35|-2  |-16.6|33.2         |4        |
2|42|-1  |-9.6 |9.6          |1        |
3|50|0   |-1.6 |0.0          |0        |
4|61|1   |9.4  |9.4          |1        |
5|70|2   |18.4 |36.8         |4        |
sum            |89.0         |10
m=89.0/10=8.9

step3:calculate the intercept (c)
c=(y+)-m(x+)
c=51.6-8.9(3)
c=24.9

the equation y=mx+c becames
y=8.9x+24.9
the value of X is gived as 10
y=8.9(10)+24.9
y=113.9
'''

'''===========================================================================================================
b--multiple linear regression multiple input feature ex( predecting house price) formula(y=b0+b1x1+b2x2+......)
================================================================================================================'''

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
x=[
    [1000,2],                      #1000 sqft 2 bedrooms
    [1500,3],                      #1500 sqft 3 bedrooms
    [2000,4],                      #2000 sqft 4 bedrooms
    [2500,4]                       #2500 sqft 4 bedrooms
    ]
y=[30,45,60,75]                   #target values (house prices)
model=LinearRegression()
model.fit(x,y)

y_pred=model.predict(x)            #predication for traning the data->for metrics
print("predication:",y_pred)             
print("MAE:",mean_absolute_error(y,y_pred))   #finding the of MAE,MSE,R2
print("MSE:",mean_squared_error(y,y_pred))    #MAE and MSE  e- show error is not 
print("R2:",r2_score(y,y_pred))              #R2=1 perfect fit

a=int(input("Enter The Squarefit:"))
b=int(input("Enter The Bedrooms:"))
z=model.predict([[a,b]]) #type:ignore
'''======================================================================================================================
c--polynomial regression:it is a type of regression analysis in which the relationship between the independent variable x 
and the dependent variable y is modelled as an nth degree polynomial.
========================================================================================================================='''
#formula:y=b0+b1x+b2x^2+......+bnx^n
'''linear vs polynomial 
linear regression                polynomial regression
y=mx+c                           y=b0+b1x+b2x^2+......+
'''

from PIL import Image                              #image processing library install by pip install pillow
imag=Image.open("images.jpg")                      #the image show the difference between linear and polynomial regression
imag.show()                                      



from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression    #polynomialfeatures and linearregression work together in polynomialfeatures 
import matplotlib.pyplot as plt
import numpy as np
c=[[1],[2],[3],[4],[5]]                    #study hours
d=[20,35,55,78,95]                         #student marks
poly=PolynomialFeatures(degree=2)          #degree of polynomial it means y=b0+b1x+b2x^2(general formula y=b0+b1x+b2x^2+......+bnx^n)
x_poly=poly.fit_transform(c) #type:ignore  #original x[1,2,3,4,5] it transform to [[1,1,1],[1,2,4],[1,3,9],[1,4,16],[1,5,25]] it means [x^0,x^1,x^2](general formula [x^0,x^1,x^2,......,x^n])
model=LinearRegression()                  
model.fit(x_poly,d)
e=int(input("Enter The Study Hours:"))
prediction=model.predict(poly.transform([[e]]))
x_graph=np.linspace(1,6,100).reshape(-1,1)      
x_graph_poly=poly.transform(x_graph)
y_graph=model.predict(x_graph_poly)
plt.scatter(c,d,label="Actual Data")

plt.plot(x_graph,y_graph,label="Polynomial Regression")

plt.xlabel("Study Hours")
plt.ylabel("Student Marks")
plt.title("Polynomial")
plt.legend()
plt.grid(True)
plt.show()



'''maths solution: y=b0+b1x+b2x^2
x=[1,2,3,4,5] y=[20,35,55,78,95]
to find b0,b1,b2 we use the formula b=(X^T*X)^-1*X^T*y
b0=2.9 b1=16.3 b2=0.5 x=0,x1=36 by using matrix form to slove it
M=|1 1 1 |   y=|20|
  |1 2 4 |     |35|
  |1 3 9 |     |55|
  |1 4 16|     |78|
  |1 5 25|     |95|
  b=(M^T*M)^-1(*M^T*y)
  M^T*M=|5 15 55   |           to caculate M^T*y first calculate semesan of y,xy,x^2*y M^T*y=|20+35+55+78+95=283             |
        |15 55 225 |                                                                         |1*20+2*35+3*55+4*78+5*95=1042  |
        |55 225 979|                                                                         |1*20+4*35+9*55+16*78+25*95=4278|
to slove M^T*M and M^T*y we use the formula b=(M^T*M)^-1(*M^T*y) to find b0,b1,b2 by using elimination method
|5 15 55   ||b0|=|283 |
|15 55 225 ||b1|=|1042|
|55 225 979||b2|=|4278|
5b0+15b1+55b2=283--(1)
15b0+55b1+225b2=1042--(2)   
55b0+225b1+979b2=4278--(3)
from equation 1 multiply 3 and equation 2 multiply 1 then subtract it to eliminate b0
15b0+55b1+225b2=1042--(2)
15b0+45b1+165b2=849--(1)*3
-----------------------
10b1+60b2=193--(4)

eliminate b0 from equation 1 and 3 by multiply 11 to equation 1
55b0+165b1+605b2=3113--(1)*11
55b0+225b1+979b2=4278--(3)
------------------------
60b1+374b2=1165--(5)

from equation 4 and 5 6*(4)
60b1+360b2=1158--(4)*6
60b1+374b2=1165--(5)
------------------
14b2=7
b2=0.5

from equation (4) put the value b2=0.5
10b1+60(0.5)=193
b1=16.3

from equation (1) put the value b1=16.3 b2=0.5
5b0+15b1+55b2=283--(1)
b0=2.9
y=b0+b1x+b2+x^2
y=2.9+16.3(6)+0.5(36)
y=2.9+97.8+18
y=118.7
'''