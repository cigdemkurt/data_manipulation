import pandas as pd
import seaborn as sns
from pandas import read_csv
import matplotlib.pyplot as plt

data=read_csv("SuperMarketAnalysis.csv")
data_num= data.select_dtypes(include=['float64','int64'])

data=data_num.dropna()
#to delete missing values
# print(data.head())

data_gi=data["gross income"]
print(data_gi.head())

sns.boxplot(x=data_gi)
plt.show()

#to calculate the threshold value
Q1= data_gi.quantile(0.25)
Q3= data_gi.quantile(0.75)
IQR= Q3-Q1

print(Q1,Q3)

print(IQR)

lower_limit= Q1-1.5*IQR
upper_limit=Q3+ 1.5*IQR

print(f'lower limit: {lower_limit} upper limit: {upper_limit}')

#values below the lower_limit and above the upper_limit  are outliers

#to be able to access outliers;

outliers= (data_gi < lower_limit) | (data_gi > upper_limit)

print(outliers.head())

# access values with fancy index

print(data_gi[outliers])

# Get rid of outliers with SUPRESSION

# equates values that are close to the lower limit to the lower limit, and values that are close to the upper limit to the upper limit

data_gi[data_gi > upper_limit]= upper_limit



data_gi[data_gi < lower_limit]= lower_limit

outliers_tf= (data_gi< lower_limit) | (data_gi> upper_limit)

print("number of outliers: ", outliers_tf.sum())

# number of outliers:  0

if outliers_tf.any():
    print("There are still outliers.")
else:
    print("The suppression process is successful, there are no outliers.")

# The suppression process is successful, there are no outliers.

