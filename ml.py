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
'''a--regression:it predicts a numerical value like(sales,house price,salary,stock) sub types a.simple linear regression b.multiple 
linear regression c.ploynomial regression


#first algorithm:linear regression 💫(y=mx+c) example for simple liner regression'''
from sklearn.linear_model import LinearRegression        
X=[[1],[2],[3],[4],[5]]
Y=[35,42,50,61,70]
model = LinearRegression()
model.fit(X,Y)
predication=model.predict([[10]])
print(predication)


'''b--multiple linear regression multiple input feature ex( predecting house price) formula(y=b0+b1x1+b2x2+......)'''
from sklearn.linear_model import LinearRegression
x=[
    [1000,2],
    [1500,3],
    [2000,4],
    [2500,4]
    ]
y=[30,45,60,75]
model=LinearRegression()
model.fit(x,y)
z=model.predict([[1800,3]])
print(z)