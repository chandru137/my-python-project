# from sklearn.linear_model import LinearRegression
# x=[[1],[2],[3],[4]]
# y=[2,4,6,8]
# model=LinearRegression()
# model.fit(x,y)
# predection=model.predict([[5]])
# print(predection)
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# df=pd.read_excel("Project-Management-Sample-Data.xlsx")
# # plt.plot(df['Days Required'],df['Progress'],marker="d",markerfacecolor="red")
# # plt.title("hello")
# # plt.xlabel("days")
# # plt.ylabel("progress")
# # plt.show()

# sns.barplot(x=df["Days Required"],y=df["Progress"],width=1.5)
# plt.title("hello")
# plt.show()

from sklearn.linear_model import LinearRegression
import pandas as pd 
q=pd.read_excel("Book1.xlsx")
a=q[["X"]]
b=q["Y"]
model=LinearRegression()
model.fit(a,b)
z=model.predict([[11]])
print(z[0])
