import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import MinMaxScaler
import scipy.cluster.hierarchy as shc
# from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

data = pd.read_excel("EastWestAirlines.xlsx", sheet_name='data')
# print(data.head())

# print(data.info())

# renaming the columns for our convenince
data= data.rename(columns={'ID#':'ID', 'Award?':'Award'})
# print(data.columns)

# dropping unwanted columns
mydata = data.drop(['ID'], axis=1)
print(mydata.columns)


#######################################################
#     K-Prototypes Clustering for mixed datatypes     #
#######################################################

# Scree plot or Elbow curve to find K
k = list(range(2,6))
cost = []
for i in k:
    model = KPrototypes(n_clusters=i, init = 'Cao', verbose=1)
    model.fit(mydata, categorical=[2,3,4,10])
    cost.append(model.cost_)

plt.figure(figsize=(10, 5))
plt.plot(k, cost, 'go--')
plt.xlabel('Number of Clusters')
plt.ylabel('Cost')
plt.title('Elbow Curve to find optimum K') 
plt.show()

kprototypes = KPrototypes(n_clusters=4, init = 'Cao', verbose=1)
kprototypes.fit(mydata, categorical=[2,3,4,10])
# labels = kprototypes.labels_
# print(labels)
clusters = kprototypes.predict(mydata, categorical=[2,3,4,10])
print(clusters)

# adding clusters to the original dataset
# data.insert(0, 'Cluster', clusters, allow_duplicates = False)
# print(data)

###################################
#     Hierarchical Clustering     #
###################################
# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(mydata)
print(scaled_data)

# # Calculating the distance matrix
# dist = pd.DataFrame(squareform(pdist(scaled_data), 'euclidean'))
# # print(dist)

# Dendrogram
plt.title("Dendrogram with single linkage")  
dend = shc.dendrogram(shc.linkage(scaled_data, method='single'))
plt.show()

plt.title("Dendrogram with Complete linkage")  
dend = shc.dendrogram(shc.linkage(scaled_data, method='complete'))
plt.show()

plt.title("Dendrogram with Average linkage")  
dend = shc.dendrogram(shc.linkage(scaled_data, method='average'))
plt.show()

plt.title("Dendrogram with Centroid linkage")  
dend = shc.dendrogram(shc.linkage(scaled_data, method='centroid'))
plt.show()

# Building Hierarchical Agglomerative Clustering model 
hclust = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')

# predict 
y_hclust = hclust.fit_predict(scaled_data) 
print(y_hclust)
# print(norm_mydata)

##################
#     DBSCAN     #
##################

dbmodel = DBSCAN(eps = 0.3, min_samples = 3).fit(scaled_data)
labels = dbmodel.labels_
print(labels)
