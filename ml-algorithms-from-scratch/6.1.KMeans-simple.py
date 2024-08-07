import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from matplotlib import style
style.use('ggplot')

X = np.array([[1,2],
            [1.5, 1.8],
            [5, 8],
            [8, 8],
            [1, 0.6],
            [9, 11]])
# plt.scatter(X[:, 0], X[:, 1], s=150, linewidths=5)
# plt.show()

#Define Classifier
# The number of cluster should be less than or equal to the data points. If n_clusters = number of data points then
# each point will be the centroid.
clf = KMeans(n_clusters=2) 
clf.fit(X)

centroids = clf.cluster_centers_
print(centroids)
labels = clf.labels_ # 0 or 1 for each data point. So the labels array will have len = total number of data points.
print(labels)

colors = ["g.", "r.", "c.", "b.", "k.", "o."]
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=25)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5)
plt.show()