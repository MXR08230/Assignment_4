import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

#To display all columns in the output
desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',20)

df2=pd.read_csv("./K-Mean_Dataset.csv")
print(df2.head())
X = df2.iloc[:,1:].values
#Removing Null values using "Mean"
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X)
x = imputer.transform(X)
#Using the elbow method to find a good number of clusters with the K-Means algorithm
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
#Implementing K means clustering
from sklearn.cluster import KMeans
nclusters = 2
km = KMeans(n_clusters=nclusters)
print(km.fit(x))
#Calculating the silhouette score
y_cluster_kmeans = km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print('Silhouette score:',score)

#question 3
#Try feature scaling and then apply K-Means on the scaled features. Did that improve the Silhouette score? If Yes, can you justify why

scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array)

from sklearn.cluster import KMeans
nclusters = 2
km = KMeans(n_clusters=nclusters)
print(km.fit(X_scaled))

#calcualating Silhouette score after applying scaling
y_scaled_cluster_kmeans = km.predict(X_scaled)
from sklearn import metrics
score = metrics.silhouette_score(X_scaled, y_scaled_cluster_kmeans)
print('Silhouette score after applying scaling:',score)