import pandas as pd
import seaborn as sns
import sklearn
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer



data = pd.read_csv('supermarket_sales new.csv')
#print(type(data))
data_num = data.select_dtypes(include=['float64', 'int64'])

data=data_num.dropna()

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

outliers_Up= (data_Up < lower_limit) | (data_Up > upper_limit)

print(outliers_Up.head())

# access values with fancy index

print(data_Up[outliers_Up])

# there is no outlier value in Unit price column

# if there are outlier values, the indexes could be accessed as follows
#data_Up[outliers].index


data_quantity=data['Quantity']
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

outliers_quantity= (data_quantity< lower_limit) | (data_quantity > upper_limit)

print(outliers_quantity.head())

print(data_quantity[outliers_quantity])

# there is no outlier value in data_quantity column


data_tax=data['Tax 5%']
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

outliers_tax= (data_tax< lower_limit) | (data_tax > upper_limit)

# print(outliers_tax.head())

print(data_tax[outliers_tax])


#AND SOLVE

# print(type(data_tax))

data_tax= pd.DataFrame(data_tax)
# print(data_tax.shape)
# (1000, 1)

t_data_tax=data_tax[~((data_tax<(lower_limit)) | (data_tax>(upper_limit))).any(axis=1)]
# print(t_data_tax.shape)
# (991, 1)

data_tax=t_data_tax


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

outliers_tax= (data_tax< lower_limit) | (data_tax > upper_limit)

print("number of outliers: ", outliers_tax.sum().sum())

# number of outliers:  0

if outliers_tax.any().any():
    print("There are still outliers.")
else:
    print("The suppression process is successful, there are no outliers.")

# The suppression process is successful, there are no outliers.

# MULTIVARIATE OUTLIER OBSERVATION WITH USING LOF

clf= LocalOutlierFactor(n_neighbors=20,contamination='auto')

clf.fit(data)

data_scores=clf.negative_outlier_factor_
print(data_scores[0:10])

print(np.sort(data_scores)[0:30])


treshold_value= np.sort(data_scores)[11]
# THE METHOD OF DELETION
new_data= data[data_scores > treshold_value]
print(new_data) #to access values that are not outliers

outliers_value= data[data_scores < treshold_value]
# print(outliers_value) to access values that are outliers

#THE METHOD OF SUPRESSION
suppression_value= data[data_scores==treshold_value]
#print(suppression_value)

#should replace outliers with suppression values

res= outliers_value.to_records(index=False) #i converted the dataframe to a numpy array
res[:]
# res[:] = [:] mean is all values in res
res[:]= suppression_value.to_records(index=False) #all values in the res were outliers, outliers were replaced with suppression (threshold) values
print(res)

outliers_value=pd.DataFrame(res, index= outliers_value.index)
print(outliers_value )

# i created a score for each observation unit and selected an observation unit as a threshold value for the score. I did the deletion method first and then the suppression method


            ## MISSING DATA ANALYSIS ##


print(data.isnull().sum())
# Unit price    0, Quantity      0 ,Tax 5%        0 there is no missing value in dataset
print(data.isnull().sum().sum()) # to access total missing value
print(data.isnull())
print(data[data.isnull().any(axis=1)]) # if there is at least one missing value, bring that line
#output: Index: []
print(data[data.notnull().all(axis=1)]) # bring all those that do not contain missing value

# ~~ direct deletion of missing values ~~

print(data.dropna()) # even if there is only one missing value in a line, it deletes that line completely
# data.dropna(inplace=True) #if you say inplace= True, it makes a permanent change, the deleted lines will not be returned

#  ~~ filling in the missing values with the average ~~
#print(data_Up.mean())
# data_Up.fillna(data_Up.mean()) #filled in the missing values with the average
# data_Up.fillna(0) #filled with 0
"""""
data.apply(lambda x:x.fillna(x), axis=0) apply means that the columns will be processed. it is determined what to do with the column
filled the empty values in each variable with the average of the variable itself
"""""
        # ~~ visualization of the missing data structure ~~

import missingno as msno
msno.bar(data); #visualizes the shortcomings in the data
msno.matrix(data); ## NaN values related to addiction are better understood with this graph
msno.heatmap(data); # shows the interdependencies of values as a percentage
#plt.show()

        # ~~ deleting missing observations
# data.dropna(how='all') deletion of observations with all data is empty
# data.dropna(axis=1) completely delete an observation that contains at least one missing data
# data.dropna(axis=1, how='all') deletes all column-based values that are NaN ones
        # ~~ simple methods of assigning values
# data["Quantity"].fillna(0) filling with 0
# data["Quantity"].fillna(data["Quantity"].mean()) filling with average
# data.apply(lambda x:x.fillna(x.mean()), axis=0)  ling in the values with their own averages
# data.where(pd.notna(data), data.mean(),axis="columns") fills in the empty values that it captures in the column with the average of that column
data = pd.read_csv('supermarket_sales new.csv')
#print(data.groupby("Product line")["Quantity"].mean()) # look at the quantity average of product lines
data["Quantity"].fillna(data.groupby("Product line")["Quantity"].transform("mean")) #when the fillna function was used together with the transform parameter, it specifically took the averages of the departments separately and filled in the average of that department, whichever department had a missing value.

missing_values = data.isnull().sum()
# print(missing_values) #i have also observed that there is no missing value in categorical variables

        # ~~ for categorical value
"""
data["Product line"].fillna(method="bfill") fill it with the next value
data["Product line"].fillna(method="ffill") fill it with the previous value
"""
        # ~~ STANDARDIZATION
# The values are converted into a distribution with an average of 0 and a standard deviation of 1
# In data sets containing negative values.
from sklearn import preprocessing
preprocessing.scale(data_num) #standardized all the variables, between 0 and 1
        # ~~ NORMALIZATION
# Especially in data sets containing positive values.
#between [0, 1] or [-1, 1]
preprocessing.normalize(data_num)
        # ~~ MIN MAX
# Convert to desired range
scaler = preprocessing.MinMaxScaler(feature_range=(10,20))
scaler.fit_transform(data_num)
        # VARIABLE TRANSFORMERS (Converting categorical variables to numerical variables)
# a) 0-1
from sklearn.preprocessing import LabelEncoder
lbe= LabelEncoder()
data["new_Gender"]= lbe.fit_transform(data["Gender"]) #For example, it can be done for gender. There are two variables 1 and 0- 0 for female, 1 for male

data["Gender"] = data["Gender"].astype("category")
data["n_gender"]= data["Gender"].cat.codes #Important Note: The gender column must have been converted to an existing categorical data, difference from line 260
print(data.head())

# b) 1 and others(0)
data["new_City"]= np.where(data["City"].str.contains("Yangon"),1,0) # only made Yangon 1, other cities 0

# c) multiclass conversion
data["class_City"]= lbe.fit_transform(data["City"])
print(data)
# d) one-hot and dummy variable trap
data_one_hot= pd.get_dummies(data,columns = ["Gender"], prefix= ["G"]) # prefix determines what the prefix will be
print(data_one_hot.head())
        #data standardization (transformation)
# d)binarize
binarizer= preprocessing.Binarizer(threshold=0.5).fit(data_num)
binarizer.transform(data_num) #If a value is greater than or equal to the threshold value, it is set to 1. If a value is less than the threshold value, it is set to 0.

        # Converting a continuous variable to a categorical variable
discretized_data=KBinsDiscretizer(n_bins=[3,2,2], encode='ordinal',strategy="quantile").fit_transform(data_num) #quantile tries to put an equal number of samples (thousands) into each category. For example, even if the data distribution is unbalanced, each bin contains as equal a number of data as possible. This is especially useful in unbalanced data sets.
print(discretized_data)

        # converting variable to index, index to variable
data["index"]= data.index
print(data.head())
# data preprocessing finished
