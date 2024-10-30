import pandas as pd
import seaborn as sns
from pandas import read_csv
import matplotlib.pyplot as plt

data=read_csv("SuperMarketAnalysis.csv")
data_num= data.select_dtypes(include=['float64','int64'])

data=data_num.dropna()
#to delete missing values
# print(data.head())

data_Up=data["Unit price"]
print(data_Up.head())

sns.boxplot(x=data_Up)
plt.show()

#to calculate the threshold value
Q1= data_Up.quantile(0.25)
Q3= data_Up.quantile(0.75)
IQR= Q3-Q1

print(Q1,Q3)

print(IQR)

lower_limit= Q1-1.5*IQR
upper_limit=Q3+ 1.5*IQR

print(f'lower limit: {lower_limit} upper limit: {upper_limit}')

#values below the lower_limit and above the upper_limit  are outliers

#to be able to access outliers;

outliers= (data_Up < lower_limit) | (data_Up > upper_limit)

print(outliers.head())

# access values with fancy index

print(data_Up[outliers])

# there is no outlier value

# if there are outlier values, the indexes could be accessed as follows
#data_Up[outliers].index
