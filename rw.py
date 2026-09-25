import pandas as pd                   #first instal the package as :(pip install pandas) commend
df = pd.DataFrame({
    'points': [88, 92, 85, 92, 95],
    'gdp' : [4,5,6,3,5],
    'country': ['US', 'France', 'US', 'Italy', 'France']
})

# Quick statistical overview
print(df['points'].describe())

 #Frequency of categories
print(df['country'].value_counts())
print(df['points'].mean())  # Mean of points
print(df['points'].median())  # Median of points
print(df['country'].mode())  # Mode of country
print(df['points'].std())  # Standard deviation of points
print(df['points'].var())  # Variance of points
print(df['points'].unique())  # Unique values in points
#maping function to transform data
points_mean = df['gdp'].mean() #we assgin mean of the gdp to the points_mean
print(df['points'].map(lambda p: p - points_mean)) #df.map(lambda p:p-->is the keyword map them 
#by using simple methode df['points']-points_mean
#df.map(lambda p:p in more data its calculate faster for each data
#operation in pandas df['points']+5 like simalar basics
print(df['gdp'].astype(str)) #changing the data type
#-------------------------------------------------------------------------------------#
#                     READING AND WRITING THE FUNTION📂
#-------------------------------------------------------------------------------------#
xy=pd.read_csv('r.csv')            # reading the funtion from csv file by the .read_csv
print(xy.iloc[0,2])                #accesing element by the index keyword .iloc[row,column]
print(xy.columns)                  #print only column
print(xy.loc[:,'age'])             #print the row value
xy['salary']=[10,20,30]            #adding the column
print(xy) 
print(xy.info())                   #information about columns and rows and emapty value 