# Naive Bayes Classifier

## What is a classifier?
A classifier is a machine learning model that is used to discriminate different objects based on certain features.


## Principle of Naive Bayes Classifier:
A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. The crux of the classifier is based on the Bayes theorem.

## Bayes Theorem:
![image](https://user-images.githubusercontent.com/89921883/226667253-79d8afdd-3d61-47be-bc1b-560e7cf750a5.png)


Using Bayes theorem, we can find the probability of A happening, given that B has occurred. Here, B is the evidence and A is the hypothesis. The assumption made here is that the predictors/features are independent. That is presence of one particular feature does not affect the other. Hence it is called naive.


According to this example, Bayes theorem can be rewritten as:

![image](https://user-images.githubusercontent.com/89921883/226667714-9b5a1757-10bf-458c-b4c5-a5034e9c79e9.png)


The variable y is the class variable(play golf), which represents if it is suitable to play golf or not given the conditions. Variable X represent the parameters/features.


X is given as,

![image](https://user-images.githubusercontent.com/89921883/226668221-974cafb8-8494-4dcb-921c-c9644d3ab7fb.png)


Here x_1,x_2….x_n represent the features, i.e they can be mapped to outlook, temperature, humidity and windy. By substituting for X and expanding using the chain rule we get,

![image](https://user-images.githubusercontent.com/89921883/226668656-7f2c1154-6374-4a73-87ff-ba8f63734711.png)


Now, you can obtain the values for each by looking at the dataset and substitute them into the equation. For all entries in the dataset, the denominator does not change, it remain static. Therefore, the denominator can be removed and a proportionality can be introduced.


![image](https://user-images.githubusercontent.com/89921883/226668880-100baadb-9613-4547-acfc-aac6da3b112d.png)


In our case, the class variable(y) has only two outcomes, yes or no. There could be cases where the classification could be multivariate. Therefore, we need to find the class y with maximum probability.

![image](https://user-images.githubusercontent.com/89921883/226669116-730e1ff5-10ed-4df6-aee5-50952b1c2eda.png)


## Gaussian Naive Bayes:
When the predictors take up a continuous value and are not discrete, we assume that these values are sampled from a gaussian distribution.
Since the way the values are present in the dataset changes, the formula for conditional probability changes to,

![image](https://user-images.githubusercontent.com/89921883/226669550-535799bb-734c-4ef6-9ee5-52f17275110b.png)
