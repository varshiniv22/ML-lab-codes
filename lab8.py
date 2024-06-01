from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import mglearn

#from matplotlib import pyplot as plt
iris=load_iris()
X=iris.data
# X, _ = make_blobs(n_samples=1000, centers=3, n_features=2)
# print(X)

kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)
print("data points belongs to clusters ",kmeans.labels_)
print("cluster centroids are as follows",kmeans.cluster_centers_)

new_data,label = make_blobs(n_samples=1, centers=1, n_features=4)
print("new data points", new_data)
print("new clusters belong to ",kmeans.predict(new_data))