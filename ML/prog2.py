import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data=pd.read_csv("housing.data")


numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
print(f"Numerical features: {list(numerical_features)}")


correlation = data[numerical_features].corr()
print(correlation)


plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()


sns.pairplot(data[numerical_features], diag_kind='kde', plot_kws={'alpha': 0.7})
plt.show()