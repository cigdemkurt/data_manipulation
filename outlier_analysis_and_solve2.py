import seaborn as sns
from pandas import read_csv
import matplotlib.pyplot as plt

from outlier_analysis import outliers

data=read_csv("SuperMarketAnalysis.csv")
data_num= data.select_dtypes(include=['float64','int64'])
new_data=data_num.dropna()
#print(new_data.head())

data_sales=new_data['Sales']
#print(data_tax.head())
# to detecting outliers

# sns.boxplot(x=data_quantity)
# plt.show()

Q1= data_sales.quantile(0.25)
Q3=data_sales.quantile(0.75)
IQR=Q3-Q1


# print(Q1,Q3,IQR)

lower_limit= Q1-1.5*IQR
upper_limit=Q3+ 1.5*IQR

print(f'lower limit: {lower_limit} upper limit: {upper_limit}')

outliers_tf= (data_sales< lower_limit) | (data_sales > upper_limit)

print(outliers_tf.head())

print(data_sales[outliers_tf])

# Get rid of outliers with SUPRESSION

# equates values that are close to the lower limit to the lower limit, and values that are close to the upper limit to the upper limit

data_sales[data_sales > upper_limit]= upper_limit



data_sales[data_sales < lower_limit]= lower_limit

outliers_tf= (data_sales< lower_limit) | (data_sales > upper_limit)

print("number of outliers: ", outliers_tf.sum())

# number of outliers:  0

if outliers_tf.any():
    print("There are still outliers.")
else:
    print("The suppression process is successful, there are no outliers.")

# The suppression process is successful, there are no outliers.
