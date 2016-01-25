---
layout: post
title: 用scikit-learn 来演绎随机森林方法
category: tool
tags: Essay
keywords: python,scikit-learn
---

### 用scikit-learn 来演绎随机森林方法

作者：Ando Saabas

来源：http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/

在之前的一篇文章中，我们讨论了如何将随机森林模型转成一个"白箱子"，就像预测变量可以由一组拥有不同特征自变量的来解释。

我对此有不少需求，但不幸的是，大多数随机森林算法包（包括 scikit-learn)并没有给出树的预测路径。因此sklearn的应用需要一个补丁来展现这些路径。幸运的是，cong 0.17 dev,scikit-learn 补充了两个附加的api，使得一些问题更加方便。获得叶子node_id，并将所有中间值存储在决策树中所有节点,不仅叶节点。通过结合这些，我们有可能可以提取每个单独预测的预测路径，以及通过检查路径来分解它们。

废话少说, 代码托管在github,你可以通过  `pip install treeinterpreter`   来获取。

### 使用treeinterpreter来分解随机森林

 首先我们将使用一个简单的数据集，来训练随机森林模型。在对测试集的进行预测的同时我们将对预测值进行分解。


```python
from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

from sklearn.datasets import load_boston
boston = load_boston()
rf = RandomForestRegressor()
rf.fit(boston.data[:300], boston.target[:300])

```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
               verbose=0, warm_start=False)



任意选择两个可以产生不同价格模型的数据点。


```python
instances = boston.data[[300, 309]] #任意选择两个可以产生不同价格模型的数据点。
print "Instance 0 prediction:", rf.predict(instances[0])
print "Instance 1 prediction:", rf.predict(instances[1])
```

    Instance 0 prediction: [ 30.27]
    Instance 1 prediction: [ 22.03]


    /Users/donganlan/anaconda/lib/python2.7/site-packages/sklearn/utils/validation.py:386: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and willraise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      DeprecationWarning)
    /Users/donganlan/anaconda/lib/python2.7/site-packages/sklearn/utils/validation.py:386: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and willraise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      DeprecationWarning)


对于这两个数据点，随机森林给出了差异很大的预测值。为什么呢？我们现在可以将预测值分解成有偏差项（就是训练集的均值）和个体差异，并探究哪些特征导致了差异，并且占了多少。
 
 
我们可以简单的使用treeinterpreter中 predict 方法来处理模型和数据。


```python
prediction, bias, contributions = ti.predict(rf, instances)
#Printint out the results:
for i in range(len(instances)):
    print "Instance", i
    print "Bias (trainset mean)", bias[i]
    print "Feature contributions:"
    for c, feature in sorted(zip(contributions[i], 
                                 boston.feature_names), 
                             key=lambda x: -abs(x[0])):
        print feature, round(c, 2)
    print "-"*20 
```

     Instance 0
    Bias (trainset mean) 25.8759666667
    Feature contributions:
    RM 4.25
    TAX -1.26
    LSTAT 0.71
    PTRATIO 0.22
    DIS 0.15
    B -0.14
    AGE 0.12
    CRIM 0.12
    RAD 0.11
    ZN 0.1
    NOX -0.1
    INDUS 0.06
    CHAS 0.06
    --------------------
    Instance 1
    Bias (trainset mean) 25.8759666667
    Feature contributions:
    RM -5.81
    LSTAT 1.66
    CRIM 0.26
    NOX -0.21
    TAX -0.15
    DIS 0.13
    B 0.11
    PTRATIO 0.07
    INDUS 0.07
    RAD 0.05
    ZN -0.02
    AGE -0.01
    CHAS 0.0
    --------------------


各个特征的贡献度按照绝对值从大到小排序。我们可以从 Instance 0中（预测值较高）可以看到，大多数正效应来自RM.LSTAT和PTRATIO。在Instance 1中（预测值较低），RM实际上对预测值有着很大的负影响，而且这个影响并没有被其他正效应所补偿，因此低于数据集的均值。

但是这个分解真的是对的么？这很容易检查：偏差项和各个特征的贡献值加起来需要等于预测值。


```python
print prediction
print bias + np.sum(contributions, axis=1)
```

    [ 30.27  22.03]
    [ 30.27  22.03]


### 对更多的数据集进行对比

当对比两个数据集时，这个方法将会很有用。例如

 * 理解导致两个预测值不同的真实原因，究竟是什么导致了房价在两个社区的预测值不同 。
 
 * 调试模型或者数据，理解为什么新数据集的平均预测值与旧数据集所得到的结果不同。

举个例子，我们将剩下的房屋价格数据分成两个部分，分别计算它们的平均估计价格。


```python
ds1 = boston.data[300:400]
ds2 = boston.data[400:]
 
print np.mean(rf.predict(ds1))
print np.mean(rf.predict(ds2))
```

    22.3327
    18.8858490566


我们可以看到两个数据集的预测值是不一样的。现在来看看造成这种差异的原因：哪些特征导致了这种差异，它们分别有多大的影响。


```python
prediction1, bias1, contributions1 = ti.predict(rf, ds1)
prediction2, bias2, contributions2 = ti.predict(rf, ds2)
#We can now calculate the mean contribution of each feature to the difference.
totalc1 = np.mean(contributions1, axis=0) 
totalc2 = np.mean(contributions2, axis=0) 
```

因为误差项对于两个测试集都是相同的（因为它们来自同一个训练集），那么两者平均预测值的不同主要是因为特征的影响不同。换句话说，特征影响的总和之差应该等于平均预测值之差，这个可以很简单的进行验证。


```python
print np.sum(totalc1 - totalc2)
print np.mean(prediction1) - np.mean(prediction2)
```

    3.4468509434
    3.4468509434



最后，我们将两个数据集中各个特征的贡献打印出来，这些数的总和正好等于与预测均值的差异。


```python
for c, feature in sorted(zip(totalc1 - totalc2, 
                             boston.feature_names), reverse=True):
    print feature, round(c, 2)
```

    LSTAT 2.23
    CRIM 0.56
    RM 0.45
    NOX 0.28
    B 0.1
    ZN 0.03
    PTRATIO 0.03
    RAD 0.03
    INDUS -0.0
    CHAS -0.0
    TAX -0.01
    AGE -0.05
    DIS -0.18


### 分类树 和 森林

完全相同的方法可以用于分类树，其中可以得到各个特征对于估计类别的贡献大小。
我们可以用iris数据集做一个例子。


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
iris = load_iris()
 
rf = RandomForestClassifier(max_depth = 4)
idx = range(len(iris.target))
np.random.shuffle(idx)
 
rf.fit(iris.data[idx][:100], iris.target[idx][:100])
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=4, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)



对单个例子进行预测


```python
instance = iris.data[idx][100:101]
print rf.predict_proba(instance)
```

    [[ 0.  0.  1.]]



```python
prediction, bias, contributions = ti.predict(rf, instance)
print "Prediction", prediction
print "Bias (trainset prior)", bias
print "Feature contributions:"
for c, feature in zip(contributions[0], 
                             iris.feature_names):
    print feature, c
```

    Prediction [[ 0.  0.  1.]]
    Bias (trainset prior) [[ 0.33  0.32  0.35]]
    Feature contributions:
    sepal length (cm) [-0.04014815 -0.00237543  0.04252358]
    sepal width (cm) [ 0.  0.  0.]
    petal length (cm) [-0.13585185 -0.13180675  0.2676586 ]
    petal width (cm) [-0.154      -0.18581782  0.33981782]


我们可以看到，对预测值是第二类影响力最大的是花瓣的长度和宽度,它们对更新之前的结果有最大影响。

### 总结

对随机森林预测值的说明其实是很简单的,与线性模型难度相同。通过使用treeinterpreter (pip install treeinterpreter)，简单的几行代码就可以解决问题。


翻译：lan

来源：http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/
