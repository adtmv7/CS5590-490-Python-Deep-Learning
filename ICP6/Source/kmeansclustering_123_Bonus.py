# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Reading the data & identifying the feature to form the clusters
customer = pd.read_csv('CC.csv')
x = customer.iloc[:, 1:17]
y = customer.iloc[:, -1] #last column of data frame

#1a Computing mean of data containing null values to replace them with its mean
MeanNA = customer.loc[:, "MINIMUM_PAYMENTS"].mean()
print('Mean of Minimum Payments is ', MeanNA)
x = x.fillna(MeanNA)

#1b Elbow point computation to determine good number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

#Plotting the elbow point on graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

# Performing K-Means clustering on the data available
nclusters = 4 #This is the K in mean
km = KMeans(n_clusters=nclusters)
km.fit(x)

#2 Evaluation of the clusters and silhouette score
y_cluster_KMeans = km.predict(x)
score = metrics.silhouette_score(x, y_cluster_KMeans)
print('Silhoutee Score of the Clusters is ', score)

#3 Feature Scaling

scaler = StandardScaler()# Fit on training set only.
scaler.fit(x)

# Apply transform to both the training set and the test set.
x= scaler.transform(x)
X_scaled_array=scaler.transform(x)
X_scaled=pd.DataFrame(X_scaled_array)
x=X_scaled

##building the model
nclusters = 4 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)
score = metrics.silhouette_score(x, y_cluster_kmeans)
print('Silhoutee Score of the Clusters after Scaling is ', score)

# Standardization of the data
scaler = StandardScaler()
scaler.fit(x)

# Projecting data on reduced dimension
x_scaler = scaler.transform(x)

# Performing Principle Component Analysis (PCA)
pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)
df2 = pd.DataFrame(data=x_pca) #printdf2

# Bonus: KMeans on PCA
# Performing K-Means clustering on the PCA data
nclusters = 4
km = KMeans(n_clusters=nclusters)
km.fit(x_pca)

# Evaluation of the clusters accuracy
y_cluster_KMeans = km.predict(x_pca)
score = metrics.silhouette_score(x_pca, y_cluster_KMeans)
print('Silhoutee Score of the Clusters after applying Kmean on PCA is ', score)