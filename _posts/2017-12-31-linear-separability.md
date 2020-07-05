---
toc: true
layout: post
description: Methods for Testing Linear Separability in Python
categories: [markdown]
title: Linear Separability
---

# Linear vs Non-Linear Classification

Two subsets are said to be linearly separable if there exists a hyperplane that separates the elements of each set in a way that all elements of one set resides on the opposite side of the hyperplane from the other set. In 2D plotting, we can depict this through a separation line, and in 3D plotting through a hyperplane.

By definition Linear Separability is defined:

Two sets $H = \{ H^1,\cdots,H^h \} \subseteq \mathbb{R}^d$ and $M = \{ M^1,\cdots,M^m \} \subseteq \mathbb{R}^d$ are said to be linearly separable if $\exists a \in \mathbb{R}^n$, $b \in \mathbb{R} : H \subseteq \{ x \in \mathbb{R}^n : a^T x > b \}$ and $M \subseteq \{ x \in \mathbb{R}^n : a^Tx \leq b \}$ [1](http://www.tarekatwan.com/wp-admin/post.php?post=102&action=edit#fn-102-1)

In simple words, the expression above states that H and M are linearly separable if there exists a hyperplane that completely separates the elements of [latex]H [/latex] and elements of $M$.

<img src="http://www.tarekatwan.com/wp-content/uploads/2017/12/linear_sep-1024x419.png" alt="" width="1000" height="409" class="alignnone size-large wp-image-202" />
<em>Image source from Sebastian Raschka</em> [^2]

In the figure above, (A) shows a linear classification problem and (B) shows a non-linear classification problem. In (A) our decision boundary is a linear one that completely separates the blue dots from the green dots. In this scenario several linear classifiers can be implemented.

In (B) our decision boundary is non-linear and we would be using non-linear kernel functions and other non-linear classification algorithms and techniques.

Generally speaking, in Machine Learning and before running any type of classifier, it is important to understand the data we are dealing with to determine which algorithm to start with, and which parameters we need to adjust that are suitable for the task. This brings us to the topic of linear separability and understanding if our problem is linear or non-linear.

As states above, there are several classification algorithms that are designed to separate the data by constructing a linear decision boundary (hyperplane) to divide the classes and with that comes the assumption: that the data is linearly separable. Now, in real world scenarios things are not that easy and data in many cases may not be linearly separable and thus non-linear techniques are applied. Without digging too deep, the decision of linear vs non-linear techniques is a decision the data scientist need to make based on what they know in terms of the end goal, what they are willing to accept in terms of error, the balance between model complexity and generalization, bias-variance tradeoff ..etc.

This post was inspired by research papers on the topic of linear separability including <u>The Linear Separability Problem: Some Testing Method</u> [^3], [^4]

My goal in this post is to apply and test few techniques in python and demonstrate how they can be implemented. Some of those techniques for testing linear separability are:

* [Domain Knowledge and Expertise](#1)
* [Data Visualization](#2)
* [Computational Geometry (Convex Hulls)](#3)
* [Linear Programming](#4)
* [Machine Learning:](#5)
      * [Perceptron](#6)
      * [Support Vector Machine](#7)

## Domain Knowledge/Expertise

It should be a no-brainer that the first step should always be to seek insight from analysts and other data scientists who are already dealing with the data and familiar with it. It is critical before embarking on any data discovery journey to always start by asking questions to better understand the purpose of the task (your goal) and gain early insight into the data from the domain experts (business data users , data/business analysts or data scientists) that are closer to the data and deal with it daily.

## Getting our data

For the other four (4) approaches listed above, we will explore these concepts using the classic [Iris data set](https://archive.ics.uci.edu/ml/datasets/iris) and implement some of the theories behind testing for linear separability using Python.

Since this is a well known data set we know in advance which classes are linearly separable (domain knowledge/past experiences coming into play here).For our analysis we will use this knowledge to confirm our findings.

> The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.

Let's get things ready first by importing the necessary libraries and loading our data.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
data = datasets.load_iris()

#create a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = pd.DataFrame(data.target)
df.head()

```



|      | sepal Length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | Target |
| :--: | :---------------: | :--------------: | :---------------: | :--------------: | :----: |
|  0   |        5.1        |       3.5        |        1.4        |       0.2        |   0    |
|  1   |        4.9        |       3.0        |        1.4        |       0.2        |   0    |
|  2   |        4.7        |       3.2        |        1.3        |       0.2        |   0    |
|  3   |        4.6        |       3.1        |        1.5        |       0.2        |   0    |
|  4   |        5.0        |       3.6        |        1.4        |       0.2        |   0    |

## Data Visiualization

The simplest and quickest method is to visualize the data. This approach may not be feasible or as straight forward if the number of features is large, making it hard to plot in 2D . In such a case, we can use a Pair Plot approach, and pandas gives us a great option to do so with `scatter_matrix`:



```python
from pandas.tools.plotting import scatter_matrix
scatter_matrix(df.iloc[:,0:4], figsize=(15,11))

```

<img class="alignnone size-full wp-image-86" src="http://www.tarekatwan.com/wp-content/uploads/2017/12/fig1.png" alt="" width="888" height="649" />

The scatter matrix above is a pair-wise scatter plot for all features in the data set (we have four features so we get a 4x4 matrix). The scatter matrix provides insight into how these variables are correlated. Let's expand upon this by creating a scatter plot for the Petal Length vs Petal Width from the scatter matrix.

```python
plt.clf()
plt.figure(figsize=(10,6))
plt.scatter(df.iloc[:,2], df.iloc[:,3])
plt.title('Petal Width vs Petal Length')
plt.xlabel(data.feature_names[2])
plt.ylabel(data.feature_names[3])
plt.show()
```



<img class="alignnone size-full wp-image-87" src="http://www.tarekatwan.com/wp-content/uploads/2017/12/fig2.png" alt="" width="612" height="387" />

It's still not that helpful. Let's color each class and add a legend so we can understand what the plot is trying to convey in terms of data distribution per class and determine if the classes can be linearly separable visually.

Let's update our code:

```python
plt.clf()
plt.figure(figsize = (10, 6))
names = data.target_names
colors = ['b','r','g']
label = (data.target).astype(np.int)
plt.title('Petal Width vs Petal Length')
plt.xlabel(data.feature_names[2])
plt.ylabel(data.feature_names[3])
for i in range(len(names)):
    bucket = df[df['Target'] == i]
    bucket = bucket.iloc[:,[2,3]].values
    plt.scatter(bucket[:, 0], bucket[:, 1], label=names[i])
plt.legend()
plt.show()
```



<img class="alignnone size-full wp-image-88" src="http://www.tarekatwan.com/wp-content/uploads/2017/12/fig3.png" alt="" width="612" height="387" />

Much better. We just plotted the entire data set, all 150 points. There are 50 data points per class. And Yes, at first glance we can see that the blue dots (Setosa class) can be easily separated by drawing a line and segregate it from the rest of the classes. But what about the other two classes?

Let's examine another approach to be more certain.

## Computational Geometry

In this approach we will use a **[Convex Hull](https://en.wikipedia.org/wiki/Convex_hull)** to check whether a particular class is linearly separable or not from the rest. In simplest terms, the convex hull represents the outer boundaries of a group of data points (classes) which is why sometimes it's called the convex envelope.

The logic when using convex hulls when testing for linear separability is pretty straight forward which can be stated as:

> Two classes X and Y are LS (Linearly Separable) if the intersection of the convex hulls of X and Y is empty, and NLS (Not Linearly Separable) with a non-empty intersection.

A quick way to see how this works is to visualize the data points with the convex hulls for each class. We will plot the hull boundaries to examine the intersections visually. We will be using the **Scipy** library to help us compute the convex hull. For more information please refer to [Scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html).

Let's update the previous code to include convex hulls.

```python
from scipy.spatial import ConvexHull

plt.clf()
plt.figure(figsize = (10, 6))
names = data.target_names
label = (data.target).astype(np.int)
colors = ['b','r','g']
plt.title('Petal Width vs Petal Length')
plt.xlabel(data.feature_names[2])
plt.ylabel(data.feature_names[3])
for i in range(len(names)):
    bucket = df[df['Target'] == i]
    bucket = bucket.iloc[:,[2,3]].values
    hull = ConvexHull(bucket)
    plt.scatter(bucket[:, 0], bucket[:, 1], label=names[i])
    for j in hull.simplices:
        plt.plot(bucket[j,0], bucket[j,1], colors[i])
plt.legend()
plt.show()
```



And our output should look like this:

<img class="alignnone size-full wp-image-89" src="http://www.tarekatwan.com/wp-content/uploads/2017/12/fig4.png" alt="" width="612" height="387" />

It is more obvious now, visually at least, that Setosa is a linearly separable class form the other two. In other words, we can easily draw a straight line to separate Setosa from non-Setosa (Setosas vs. everything else). Both Versicolor and Virginica classes are not linearly separable because we can see there is indeed an intersection.

## Linear Programming

By definition Linear Separability is defined:

Two sets $H = \{ H^1,\cdots,H^h \} \subseteq \mathbb{R}^d$ and $M = \{ M^1,\cdots,M^m \} \subseteq \mathbb{R}^d$ are said to be linearly separable if $\exists a \in \mathbb{R}^n$, $b \in \mathbb{R} : H \subseteq \{ x \in \mathbb{R}^n : a^T x \gt; b \}$  and $M \subseteq \{ x \in \mathbb{R}^n : a^Tx \leq b \}$ [^4]

In simple words, the expression above states that H and M are linearly separable if there exists a hyperplane that completely separates the elements of $H$ and elements of $M$.

$H$ and $M$ are linearly separable if the optimal value of Linear Program $(LP)$ is $0$

Here is a great post that implements this in R which I followed as an inspiration for this section on linear programming with python: [Testing for Linear Separability with LP in R](https://www.joyofdata.de/blog/testing-linear-separability-linear-programming-r-glpk/) [^5].

Below is the code in python using scipy `linprog(method='simplex')` to solve our linear programming problem. If we examine the output, using LP (Linear Programming) method we can conclude that it is possible to have a hyperplane that linearly separates Setosa from the rest of the classes, which is the only linearly separable class from the rest.

If the problem is solvable, the Scipy output will provide us with additional information:

| Returns            |                                          |
| ------------------ | ---------------------------------------- |
| **success**: bool  | True or False (True if a solution was found) |
| **status**: int    | **0** : Optimization terminated successfully, **1** : Iteration limit reached, **2** : Problem appears to be infeasible, **3** : Problem appears to be unbounded |
| **message** : str  | Describing the status                    |
| **x**: ndarray     | The independent variable vector which optimizes the linear programming problem. |
| **slack**: ndarray | The values of the slack variables. Each slack variable corresponds to an inequality constraint. If the slack is zero, then the corresponding constraint is active. |
| **nit** : int      | The number of iterations performed.      |
| **fun** : float    | Value of the objective function          |

For our example, I am only looking at the status/success to determine if a solution was found or not.



```python
from scipy.optimize import linprog
 dic = {0: 'setosa', 1: 'versicolor', 2: 'verginica'}
 
for i in dic.keys():
    df["newTarget"] = np.where(df['Target'] == i, 1 , -1)
     
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    tmp = df.iloc[:,[2,3]].values
    tmp = sc.fit_transform(tmp)
 
    xx = np.array(df.newTarget.values.reshape(-1,1) * tmp)
    t = np.where(df['Target'] == i, 1 , -1)
     
    #2-D array which, when matrix-multiplied by x, gives the values of 
    #the upper-bound inequality constraints at x.
    A_ub = np.append(xx, t.reshape(-1,1), 1)
     
    #1-D array of values representing the upper-bound of each 
    #inequality constraint (row) in A_ub.
    b_ub = np.repeat(-1, A_ub.shape[0]).reshape(-1,1)
     
    # Coefficients of the linear objective function to be minimized.
    c_obj = np.repeat(1, A_ub.shape[1])
    res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub,
                  options={"disp": False})
     
    if res.success:
        print('There is linear separability between {} and the rest'.format(dic[i]))
    else:
        print('No linear separability between {} and the rest'.format(dic[i]))
```
```
>>>output
There is linear separability between setosa and the rest
No linear separability between versicolor and the rest
No linear separability between verginica and the rest
```



## Machine Learning

In this section we will examine two classifiers for the purpose of testing for linear separability: the **[Perceptron](https://en.wikipedia.org/wiki/Perceptron)** (simplest form of Neural Networks) and **[Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine)** (part of a class known as [Kernel Methods](https://en.wikipedia.org/wiki/Kernel_method))


### Single Layer Perceptron

The **perceptron** is an algorithm used for binary classification and belongs to a class of linear classifiers. In binary classification, we are trying to separate data into two buckets: either you are in Buket A or Bucket B. This can be stated even simpler: either you are in Bucket A or not in Bucket A (assuming we have only two classes) and hence the name binary classification.
$$
\large \begin{cases} \displaystyle 1 &\text {if w . x + b} > {0}\\ 0 &\text {otherwise} \end{cases}
$$


A single layer perceptron will only converge if the input vectors are linearly separable. In this state, all input vectors would be classified correctly indicating linear separability. It will not converge if they are not linearly separable. In other words, it will not classify correctly if the data set is not linearly separable. For our testing purpose, this is exactly what we need.

We will apply it on the entire data instead of splitting to test/train since our intent is to test for linear separability among the classes and not to build a model for future predictions.

We will use Scikit-Learn and pick the Perceptron as our linear model selection. Before that, let's do some basic data preprocessing tasks:

```python
# Data Preprocessing
x = df.iloc[:, [2,3]].values
# we are picking Setosa to be 1 and all other classes will be 0
y = (data.target == 0).astype(np.int) 

#Perform feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x = sc.fit_transform(x)
```

Now, let's build our classifier:

```python
from sklearn.linear_model import Perceptron
perceptron = Perceptron(random_state = 0)
perceptron.fit(x, y)
predicted = perceptron.predict(x)
```

To get a better intuition on the results we will plot the confusion matrix and decision boundary.

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, predicted)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('Perceptron Confusion Matrix - Entire Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()
```



<img class="alignnone size-full wp-image-91" src="http://www.tarekatwan.com/wp-content/uploads/2017/12/fig6.png" alt="" width="301" height="307" />

Now, let's draw our decision boundary:

```python
from matplotlib.colors import ListedColormap
plt.clf()
X_set, y_set = x, y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, perceptron.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('navajowhite', 'darkkhaki')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Perceptron Classifier (Decision boundary for Setosa vs the rest)')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.show()
```



<img class="alignnone size-full wp-image-92" src="http://www.tarekatwan.com/wp-content/uploads/2017/12/fig7.png" alt="" width="409" height="278" />

We can see that our Perceptron did converge and was able to classify Setosa from Non-Setosa with perfect accuracy because indeed the data is linearly separable. This would not be the case if the data was not linearly separable. So, let's try it on another class.

Outputs below are for Versicolor class:

<img class="alignnone size-full wp-image-93" src="http://www.tarekatwan.com/wp-content/uploads/2017/12/fig8.png" alt="" width="310" height="307" /> <img class="alignnone size-full wp-image-94" src="http://www.tarekatwan.com/wp-content/uploads/2017/12/fig9.png" alt="" width="418" height="278" />



### Support Vector Machines

Now, let's examine another approach using **Support Vector Machines (SVM)** with a **linear kernel**. In order to test for Linear Separability we will pick a hard-margin (for maximum distance as opposed to soft-margin) SVM with a linear kernel. Now, if the intent was to train a model our choices would be completely different. But, since we are testing for linear separability, we want a rigid test that would fail (or produce erroneous results if not converging) to help us better assess the data at hand.

<img class="alignnone size-full wp-image-101" src="http://www.tarekatwan.com/wp-content/uploads/2017/12/svm.png" alt="" width="400" height="431" />
<em>Image source Wikipedia: Maximum-margin hyperplane</em> [^6]

Now, let's code:

```python
from sklearn.svm import SVC
svm = SVC(C=1.0, kernel='linear', random_state=0)
svm.fit(x, y)

predicted = svm.predict(x)

cm = confusion_matrix(y, predicted)

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('SVM Linear Kernel Confusion Matrix - Setosa')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
```



Here are the plots for the confusion matrix and decision boundary:

<img class="alignnone size-full wp-image-95" src="http://www.tarekatwan.com/wp-content/uploads/2017/12/fig10.png" alt="" width="323" height="307" /> <img class="alignnone size-full wp-image-96" src="http://www.tarekatwan.com/wp-content/uploads/2017/12/fig11.png" alt="" width="421" height="278" />

Perfect separartion/classification indicating a linear separability.

Now, let's examin and rerun the test against Versicolor class and we get the plots below. Interesting enough, we don't see a decision boundary and the confusion matrix indicates the classifier is not doing a good job at all.

<img class="alignnone size-full wp-image-97" src="http://www.tarekatwan.com/wp-content/uploads/2017/12/fig12.png" alt="" width="332" height="307" /> <img class="alignnone size-full wp-image-98" src="http://www.tarekatwan.com/wp-content/uploads/2017/12/fig13.png" alt="" width="440" height="278" />

Now, for fun and to demonstrate how powerful SVMs can be let's apply a non-linear kernel. In this case we will apply a Gaussian Radial Basis Function known as RBF Kernel. A slight change to the code above and we get completely different results:

```python
x = df.iloc[:, [2,3]].values
y = (data.target == 1).astype(np.int) # we are picking Versicolor to be 1 and all other classes will be 0

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0)
svm.fit(x, y)

predicted = svm.predict(x)

cm = confusion_matrix(y, predicted)

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('SVM RBF Confusion Matrix - Versicolor')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
```



<img class="alignnone size-full wp-image-99" src="http://www.tarekatwan.com/wp-content/uploads/2017/12/fig14.png" alt="" width="304" height="307" /> <img class="alignnone size-full wp-image-100" src="http://www.tarekatwan.com/wp-content/uploads/2017/12/fig15.png" alt="" width="426" height="278" />

Hope this helps.

**References**:

[^1]: Sebastian Raschka - Naive Bayes and Text Classification http://sebastianraschka.com/Articles/2014_naive_bayes_1.html
[^2]: The Linear Separability Problem: Some Testing Methods http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.121.6481&rep=rep1&type=pdf
[^3]: A Simple Algorithm for Linear Separability Test http://mllab.csa.iisc.ernet.in/downloads/Labtalk/talk30_01_08/lin_sep_test.pdf
[^4]: Convex Optimization, Linear Programming: http://www.stat.cmu.edu/~ryantibs/convexopt-F13/scribes/lec2.pdf
[^5]: Test for Linear Separability with Linear Programming in R https://www.joyofdata.de/blog/testing-linear-separability-linear-programming-r-glpk/
[^6]: Support Vector Machine https://en.wikipedia.org/wiki/Support_vector_machine