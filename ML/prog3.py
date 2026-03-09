import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Perform PCA to reduce dimensionality to 2 components
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(iris_df)

# Create a DataFrame for the reduced data
reduced_df = pd.DataFrame(data_reduced, columns=['PC 1', 'PC 2'])
reduced_df['target'] = iris.target

# Plot the reduced data
colors = ['r', 'g', 'b']
target_names = iris.target_names

for i, label in enumerate(np.unique(reduced_df['target'])):
    plt.scatter(
        reduced_df[reduced_df['target'] == label]['PC 1'],
        reduced_df[reduced_df['target'] == label]['PC 2'],
        label=target_names[label],
        color=colors[i]
    )

plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.show()