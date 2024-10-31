import seaborn as sns
from pandas import read_csv
import matplotlib.pyplot as plt
import pandas as pd

from outlier_analysis import outliers

data=read_csv("SuperMarketAnalysis.csv")
data_num= data.select_dtypes(include=['float64','int64'])
new_data=data_num.dropna()
# print(new_data.head())

data_tax=new_data['Tax 5%']
#print(data_tax.head())
# to detecting outliers

# sns.boxplot(x=data_quantity)
# plt.show()

Q1= data_tax.quantile(0.25)
Q3=data_tax.quantile(0.75)
IQR=Q3-Q1


# print(Q1,Q3,IQR)

lower_limit= Q1-1.5*IQR
upper_limit=Q3+ 1.5*IQR

# print(f'lower limit: {lower_limit} upper limit: {upper_limit}')

outliers= (data_tax< lower_limit) | (data_tax > upper_limit)

# print(outliers.head())

# print(data_tax[outliers])


#AND SOLVE

# print(type(data_tax))

data_tax= pd.DataFrame(data_tax)
# print(data_tax.shape)
# (1000, 1)

t_data_tax=data_tax[~((data_tax<(lower_limit)) | (data_tax>(upper_limit))).any(axis=1)]
# print(t_data_tax.shape)
# (991, 1)

# data_tax=t_data_tax


#this means there are 8 outlier value in data_tax
# deletion process is completed

# -OR ELIMINATE OUTLIERS BY FILLING IN WITH THE AVERAGE-

# if outliers do not want to be deleted from the data set

# new_values= data_tax[outliers]=data_tax.mean()
# data_tax= new_values
# print(data_tax.head())


# - OR SUPRESSION

# equates values that are close to the lower limit to the lower limit, and values that are close to the upper limit to the upper limit

data_tax[data_tax > upper_limit]= upper_limit



data_tax[data_tax < lower_limit]= lower_limit

outliers= (data_tax< lower_limit) | (data_tax > upper_limit)

print("number of outliers: ", outliers.sum().sum())

# number of outliers:  0

if outliers.any().any():
    print("There are still outliers.")
else:
    print("The suppression process is successful, there are no outliers.")

# The suppression process is successful, there are no outliers.
