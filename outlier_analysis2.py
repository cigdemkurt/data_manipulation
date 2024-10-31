import seaborn as sns
from pandas import read_csv
import matplotlib.pyplot as plt

from outlier_analysis import outliers

data=read_csv("SuperMarketAnalysis.csv")
data_num= data.select_dtypes(include=['float64','int64'])
new_data=data_num.dropna()
print(new_data.head())

data_quantity=new_data['Quantity']
print(data_quantity.head())
# to detecting outliers

# sns.boxplot(x=data_quantity)
# plt.show()

Q1= data_quantity.quantile(0.25)
Q3=data_quantity.quantile(0.75)
IQR=Q3-Q1


print(Q1,Q3,IQR)

lower_limit= Q1-1.5*IQR
upper_limit=Q3+ 1.5*IQR

print(f'lower limit: {lower_limit} upper limit: {upper_limit}')

outliers= (data_quantity< lower_limit) | (data_quantity > upper_limit)

print(outliers.head())

print(data_quantity[outliers])

# there is no outlier value in data_quantity column






