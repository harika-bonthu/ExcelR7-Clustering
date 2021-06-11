import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import scipy.cluster.hierarchy as shc
# from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

# load the data
data = pd.read_csv("crime_data.csv")

# print(data.head())

# print(data.isnull().any()) # No null values

# dropping the categorical feature and copy the remaining data to another dataframe
mydata = data.drop(['Unnamed: 0'], axis=1)
# print(mydata.head())

# Scaling Data
scaler = MinMaxScaler()
norm_mydata = mydata.copy()
def minmaxscaler(x):
    for columnName, columnData in x.iteritems():
        x[columnName] = scaler.fit_transform(np.array(columnData).reshape(-1, 1))
    
minmaxscaler(norm_mydata)
# print(norm_mydata.head())

###################################
#     Hierarchical Clustering     #
###################################

# # Calculating the distance matrix
# dist = pd.DataFrame(squareform(pdist(norm_mydata), 'euclidean'))
# # print(dist)

# Dendrogram
plt.title("Dendrogram with single linkage")  
dend = shc.dendrogram(shc.linkage(norm_mydata, method='single'))
plt.show()

plt.title("Dendrogram with Complete linkage")  
dend = shc.dendrogram(shc.linkage(norm_mydata, method='complete'))
plt.show()

plt.title("Dendrogram with Average linkage")  
dend = shc.dendrogram(shc.linkage(norm_mydata, method='average'))
plt.show()

plt.title("Dendrogram with Centroid linkage")  
dend = shc.dendrogram(shc.linkage(norm_mydata, method='centroid'))
plt.show()

# Building Hierarchical Agglomerative Clustering model 
hclust = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')

# predict 
y_hclust = hclust.fit_predict(norm_mydata) 
# print(y_hclust)
# print(norm_mydata)

# print(norm_mydata.iloc[y_hclust==3,0])

#PLot scatter diagram for clusters
plt.figure(figsize=(10,7))
plt.scatter(norm_mydata.iloc[y_hclust==0,0],norm_mydata.iloc[y_hclust==0,1], c='r', label='Cluster 1')
plt.scatter(norm_mydata.iloc[y_hclust==1,0],norm_mydata.iloc[y_hclust==1,1], c='g', label='Cluster 2')
plt.scatter(norm_mydata.iloc[y_hclust==2,0],norm_mydata.iloc[y_hclust==2,1], c='y', label='Cluster 3')
plt.scatter(norm_mydata.iloc[y_hclust==3,0],norm_mydata.iloc[y_hclust==3,1], c='b', label='Cluster 4')

plt.title("Clustering for crime data")
plt.xlabel("Murder Rate")
plt.ylabel("Assault Rate")
plt.legend()
plt.show()


#############################
#     KMeans Clustering     #
#############################

# Scree plot or Elbow curve to find K
k = list(range(2,11))
sum_of_squared_distances = []
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(norm_mydata)
    sum_of_squared_distances.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(k, sum_of_squared_distances, 'go--')
plt.xlabel('Number of Clusters')
plt.ylabel('Within Cluster Sum of squares')
plt.title('Elbow Curve to find optimum K') 
# plt.show()

# Building KMeans Model and predicting
# Instantiating
kmeans4 = KMeans(n_clusters = 4)

# Training the model
kmeans4.fit(norm_mydata)

# predicting
y_pred = kmeans4.fit_predict(norm_mydata)
# print(y_pred)

# Storing the y_pred values in a new column
data['Cluster'] = y_pred+1 #to start the cluster number from 1
print(data.head())

# Storing the centroids to a dataframe
centroids = kmeans4.cluster_centers_
centroids = pd.DataFrame(centroids, columns=['Murder', 'Assault', 'UrbanPop', 'Rape'])
centroids.index = np.arange(1, len(centroids)+1) # Start the index from 1
print("Centroids"+'\n',centroids)

# Sample visualization of clusters
plt.figure(figsize=(12,6))
sns.set_palette("pastel")
sns.scatterplot(x=data['Murder'], y = data['Assault'], hue=data['Cluster'], palette='bright')
# plt.show()

# Inferences
# count the number of states of each cluster
data['Cluster'].value_counts()

# Finding the means of clusters
kmeans_mean_cluster = pd.DataFrame(round(data.groupby('Cluster').mean(),1))
print(kmeans_mean_cluster)

# Cluster 2 states has the least crime rate where as Cluster 1 states has the highest Assault, Rape rate.

# View cluster 2 data
data[data['Cluster']==2]


##################
#     DBSCAN     #
##################

dbmodel = DBSCAN(eps = 0.3, min_samples = 2).fit(norm_mydata)
labels = dbmodel.labels_
print(labels)
