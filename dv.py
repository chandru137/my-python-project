import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
df=pd.read_excel('Project-Management-Sample-Data.xlsx')
#plt.plot(df["Days Required"],df["Progress"],marker='s')            #squre shape
#plt.plot(df["Days Required"],df["Progress"],marker='d')            #diamend shape
#plt.plot(df["Days Required"],df["Progress"],marker='^')            #up symbol shape
#plt.plot(df["Days Required"],df["Progress"],marker='*')            #star shape
#plt.plot(df["Days Required"],df["Progress"],marker='.')            #point shape
# plt.plot(
#     df["Days Required"],df["Progress"],marker='o',
#     markersize=10,                                                  #size of the shape
#     markerfacecolor='red',                                          #color of the shape
#     markeredgewidth=2                                               #edge width of the shape
# )
# plt.title("Task report")                                           #title of the graph📈
# plt.xlabel("Days Required")                                        #X-axies
# plt.ylabel("Progress")                                             #Y-axies
# plt.show()
#=======================================================================
#                                      BAR CHART                           #📊
#=======================================================================
# sns.barplot(data=df,x="Days Required",y="Progress",color="red",width=2) 
# plt.title("Task report")                                           
# plt.show()
#=======================================================================
#                                   HISTOGRAM                              #mathamatical expretion like graph
#=======================================================================
# sns.histplot(data=df,x="Days Required",kde=True,bins=5)                 
# plt.title("someting")
# plt.show()
#=======================================================================
#                                  scatter plot                             # graph in dot 
#=======================================================================
# sns.scatterplot(data=df,x="Days Required",y="Progress")
# plt.title("sc")
# plt.show()
#=======================================================================
#                                 correlation heatmap
#=======================================================================
# corr=df[['Days Required','Progress']].corr()
# sns.heatmap(corr,annot=True)
# plt.title("corelation")
# plt.show()