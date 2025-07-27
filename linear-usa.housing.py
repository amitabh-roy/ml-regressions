
"""
Problem Statement
A real state agents want the help to predict the house price for regions in the USA. 
He gave you the dataset to work on and you decided to use Linear Regressioon Model. 
Create a model which will help him to estimate of what the house would sell for.

Dataset contains 7 columns and 5000 rows with CSV extension. The data contains the following columns :

'Avg. Area Income': Avg. Income of householder of the city house is located in.
'Avg. Area House Age': Avg. Age of Houses in same city.
'Avg. Area Number of Rooms': Avg. Number of Rooms for Houses in same city.
'Avg. Area Number of Bedrooms': Avg. Number of Bedrooms for Houses in same city.
'Area Population': Population of city.
'Price': Price that the house sold at.
'Address': Address of the houses.

https://github.com/huzaifsayed/Linear-Regression-Model-for-House-Price-Prediction
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

import os
print(os.listdir("../data-sets/"))

# Load the dataset  
housing = pd.read_csv("../data-sets/USA_Housing.csv")
print(housing.head())
print(housing.info())
print(housing.describe())
print(housing.columns)
sns.set_style('whitegrid')
sns.pairplot(housing)
plt.show()
sns.distplot(housing['Price'])

print(housing.columns)