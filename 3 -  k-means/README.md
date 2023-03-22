# K-Means-Clustering-Algorithm-from-Scratch-in-Python

Notebook is also availble on google colab. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DE_UIYWdnKAk56DPfZGd1AK7APCgrGsL?usp=sharing)



## The Algorithm
K-Means is actually one of the simplest unsupervised clustering algorithm. Assume we have input data points x1,x2,x3,…,xn and value of K(the number of clusters needed). We follow the below procedure:

1. Pick K points as the initial centroids from the data set, either randomly or the first K.
2. Find the Euclidean distance of each point in the data set with the identified K points — cluster centroids.
3. Assign each data point to the closest centroid using the distance found in the previous step.
4. Find the new centroid by taking the average of the points in each cluster group.
5. Repeat 2 to 4 for a fixed number of iteration or till the centroids don’t change.
