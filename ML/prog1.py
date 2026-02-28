import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
housing_data=fetch_california_housing(as_frame=True)
data=housing_data.frame
data

numerical_features=data.select_dtypes(include=['float64','int64']).columns
print(f"Numerical features:{list(numerical_features)}")

plt.figure(figsize=[15,10])

for i,feature in enumerate(numerical_features):
    plt.subplot(3,3,i+1)
    plt.hist(data[feature],bins=30,color='skyblue',edgecolor='black')
    plt.title (f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


plt.figure(figsize=[15,10])
for i, feature in enumerate(numerical_features):
    plt.subplot(3,3,i+1)
    sns.boxplot(x=data[feature],color='lightgreen')
    plt.title(f"Box Plot of{feature}")
    plt.xlabel(feature)
plt.tight_layout()  
plt.show()

for feature in numerical_features:
     Q1=data[feature].quantile(0.25)
     Q2=data[feature].quantile(0.75)
     IQR=Q2-Q1
     lower_bound=Q1-1.5*IQR
     upper_bound=Q2-1.5*IQR
     outliers=data[(data[feature]<lower_bound)|(data[feature]>upper_bound)]