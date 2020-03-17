# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics
warnings.filterwarnings("ignore")

# Reading input data pertaining to Boston Housing
housingdata = pd.read_csv('trainBoston.csv')
x = housingdata.iloc[:, :14]
y = housingdata.iloc[:, -1]
print(x.shape, y.shape)

print(housingdata["medv"].value_counts())

# Correlation of numerical features associated with target
numeric_features = housingdata.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print('Features positively correlated to the target:')
print(corr['medv'].sort_values(ascending=False)[:5], '\n')
print('Features negatively correlated to the target:')
print(corr['medv'].sort_values(ascending=False)[-5:])

# Handling null values within the data set
nulls = pd.DataFrame(housingdata.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# Cluster identification
pairs = sns.pairplot(housingdata, palette='hus1')
sns.FacetGrid(housingdata, hue="medv", size=4).map(plt.scatter, "crim", "age").add_legend()
sns.FacetGrid(housingdata, hue="medv", size=4).map(plt.scatter, "crim", "nox").add_legend()
sns.FacetGrid(housingdata, hue="medv", size=4).map(plt.scatter, "crim", "dis").add_legend()
sns.FacetGrid(housingdata, hue="medv", size=4).map(plt.scatter, "crim", "ptratio").add_legend()
plt.show()

# Elbow point computation to determine optimum number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plotting the elbow point on graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

# Preprocessing the data for standardizing its distribution
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns=x.columns)

# Performing K-Means clustering on the data available
nclusters = 3
km = KMeans(n_clusters=nclusters)
km.fit(x)

# Evaluation of the clusters accuracy
y_cluster_KMeans = km.predict(x)
scoreCluster = metrics.silhouette_score(x, y_cluster_KMeans, sample_size=40)
print('Silhoutee Score of the data upon clustering for n_cluster = 3 ', scoreCluster)

# Performing K-Means clustering on the data available
nclusters = 4
km = KMeans(n_clusters=nclusters)
km.fit(x)

# Evaluation of the clusters accuracy
y_cluster_KMeans = km.predict(x)
scoreCluster = metrics.silhouette_score(x, y_cluster_KMeans, sample_size=40)
print('Silhoutee Score of the data upon clustering for n_cluster = 4 ', scoreCluster)

# Performing K-Means clustering on the data available
nclusters = 5
km = KMeans(n_clusters=nclusters)
km.fit(x)

# Evaluation of the clusters accuracy
y_cluster_KMeans = km.predict(x)
scoreCluster = metrics.silhouette_score(x, y_cluster_KMeans, sample_size=40)
print('Silhoutee Score of the data upon clustering for n_cluster = 5 ', scoreCluster)