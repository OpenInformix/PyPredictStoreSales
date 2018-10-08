
### This sample code is based on DataBriefing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

data = pd.read_csv('./data/train.csv')

tmp1 = data.head()
print("\n------ Head ---------\n", tmp1 )

# Generates descriptive statistics that summarize the central tendency, 
# dispersion and shape of a dataset’s distribution, excluding NaN values.
tmp1 = data.describe()
print("\n------- Describe ------\n", tmp1)

tmp1 = data.dtypes
print("\n-------- Type -------\n", tmp1)


# data.StateHoliday.unique()
tmp1 = data.StateHoliday.unique()
print("\n-------- StateHoliday -------\n", tmp1)

# List the number of unique values for each colum
# Apply the function to each column (axis=0)

def count_unique(column):
    return len(column.unique())

data.apply(count_unique, axis=0).astype(np.int32)

data.isnull().any()

# Filter data for store 150 and plot sales data for first 365 days
store_data = data[data.Store==150].sort_values('Date')
plt.figure(figsize=(20, 10))  # Set figsize to increase size of figure
plt.plot(store_data.Sales.values[:365])

plt.figure(figsize=(20, 10))
plt.scatter(x=store_data[data.Open==1].Promo, y=store_data[data.Open==1].Sales, alpha=0.1)
plt.xlabel('Promo')
plt.ylabel('Sales')

transformed_data = data.drop(['Store', 'Date', 'Customers'], axis=1)

transformed_data.head()
transformed_data = pd.get_dummies(transformed_data, columns=['DayOfWeek', 'StateHoliday'])

X = transformed_data.drop(['Sales'], axis=1).values
y = transformed_data.Sales.values
print("The training dataset has {} examples and {} features.".format(X.shape[0], X.shape[1]))


#########################################
# Building and cross-validating a model
##########################################

from sklearn.linear_model import LinearRegression
from sklearn import cross_validation as cv

lr = LinearRegression()
kfolds = cv.KFold(X.shape[0], n_folds=4, shuffle=True, random_state=42)
scores = cv.cross_val_score(lr, X, y, cv=kfolds)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

#########################################
# Visualizing predictions
##########################################

# We'll single out store 150 again and train our model on every store except store 150 and then predict sales for store 150.
# Remember: Always use different data for training and predicting.
lr = LinearRegression()
X_store = pd.get_dummies(data[data.Store!=150], columns=['DayOfWeek', 'StateHoliday']).drop(['Sales', 'Store', 'Date', 'Customers'], axis=1).values
y_store = pd.get_dummies(data[data.Store!=150], columns=['DayOfWeek', 'StateHoliday']).Sales.values
lr.fit(X_store, y_store)
y_store_predict = lr.predict(pd.get_dummies(store_data, columns=['DayOfWeek', 'StateHoliday']).drop(['Sales', 'Store', 'Date', 'Customers'], axis=1).values)



plt.figure(figsize=(20, 10))  # Set figsize to increase size of figure
plt.plot(store_data.Sales.values[:365], label="ground truth")
plt.plot(y_store_predict[:365], c='r', label="prediction")
plt.legend()



