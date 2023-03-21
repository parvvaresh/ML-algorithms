# KNN from scratch [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rwhhcCujhQKvo06iOiArJSnpqoCt4n_U?usp=sharing)

![image](https://user-images.githubusercontent.com/89921883/222410118-9fac734a-c4c3-4c61-bda1-b6003604bd5e.png)

**Introduction**

The reason we find that much importance is given to classification algorithms and not much is given to regression algorithms is because a lot of problems faced during our daily routine belong to the classification task. For example, we would like to know whether a tumor is malignant or benign, we would like to know whether the product we sold was received positively or negatively by the consumers, etc. K nearest neighbors is another classification algorithm and it is very simple one too. If you are following this article after K means algorithm, don't get confused as these both belong to different domains of learning. K means is a clustering/unsupervised algorithm whereas K nearest neighbors is a classification/supervised learning algorithm.



---
**What is K??**

In K means algorithm, for each test data point, we would be looking at the K nearest training data points and take the most frequently occurring classes and assign that class to the test data. Therefore, K represents the number of training data points lying in proximity to the test data point which we are going to use to find the class.


---

**K Nearest Neighbours — Pseudocode**
1. Load the training and test data 
2. Choose the value of K 
3. For each point in test data:
       - find the Euclidean distance to all training data points
       - store the Euclidean distances in a list and sort it 
       - choose the first k points 
       - assign a class to the test point based on the majority of      classes present in the chosen points
4. End 
