import pandas as pd
w=pd.read_excel('Project-Management-Sample-Data.xlsx')# for complite data
print(w)                           
# first we install the openpyxl package by using the pip install openpyxl
x=pd.read_excel('Project-Management-Sample-Data.xlsx',sheet_name='sheet1')
y=pd.read_excel('Project-Management-Sample-Data.xlsx',sheet_name=0)
z=pd.read_excel('Project-Management-Sample-Data.xlsx',sheet_name=['sheet1','sheet2'])
#first x by using sheet name
#y by using indexing
#z for specific sheet
print(w.info())                    #number of rows,number of columns,column name,data type,missing values
print(w.isnull().sum())            #checking the missing value
print(w.dropna())                    #removing missed row
