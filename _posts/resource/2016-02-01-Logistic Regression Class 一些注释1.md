---
layout: post
title: Logistic Regression Class 一些注释1
category: life
tags: Essay
keywords: Logistic Regression Class ,python
---


今天看了一下Logistic Regression Class的一些代码，获得一些编程技巧


$$
J(\theta) = - \frac{1}{m} \left[ \sum_{i=1}^{m} \sum_{j=1}^{k}  1\left\{y^{(i)} = j\right\} \log \frac{e^{\theta_j^T x^{(i)}}}{\sum_{l=1}^k e^{ \theta_l^T x^{(i)} }}\right]
$$


对于其中的$ \log \dfrac{e^{\theta_j^T x^{(i)}}}{\sum_{l=1}^k e^{ \theta_l^T x^{(i)} }} $ 我们可以用python中的一些不错的函数来表示如下：


```python
def LRcost(self,theta):
		#print self.X.shape,theta.reshape(self.classNum,self.N).shape
		theta=theta.reshape(self.classNum,self.N);   #矩阵计算 先变回来
		M=np.dot(theta,self.X)
		#print theta.reshape(self.classNum,self.N)
		M=M-M.max()
		h=np.exp(M)
		h=np.true_divide(h,np.sum(h,0))
		#print -np.sum(groundTruth*np.log(h))/self.M
		cost = -np.sum(self.groundTruth*np.log(h))/self.M+self.lam/2.0*np.sum(theta**2);     #rigde惩罚
		grad = -np.dot(self.groundTruth-h,self.X.transpose())/self.M+self.lam*theta;
		grad = grad.reshape(self.classNum*self.N)
		return cost,grad
```


```python
#其中  np.true_divide  函数完成了该功能
import numpy as np
x = np.arange(5)
np.true_divide(x, 4)   # 对每个元素除以四，在R语言中一般可以轻松表示
```




    array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])




```python
np.true_divide(x,np.sum(x))   #这里就是表达题中的需求
```




    array([ 0. ,  0.1,  0.2,  0.3,  0.4])



这里比较有意思的就是尽量用矩阵相乘来进行计算，我们可以看到，惩罚项theta应该是一列，但是有时候计算时，对于各个分类又要变成矩阵，这就涉及reshape。所以我们可以用上面的
theta=theta.reshape(self.classNum,self.N); 来变成矩阵
有时候又要从矩阵变成一个list时候：可以
self.theta = np.zeros((self.classNum,self.N)).reshape(self.classNum*self.N)  

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML">
</script>

$- \dfrac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{k}  1 \left\{ {y^{(i)} = j} \right\} $有非常有意思的表达方式，请看


```python
self.label,self.y=np.unique(y,return_inverse=True)
		self.classNum =self.label.size
		self.theta = np.zeros((self.classNum,self.N)).reshape(self.classNum*self.N)   #一列  种类数*变量个数
		self.groundTruth=np.zeros((self.classNum,self.M))
		self.groundTruth[self.y,np.arange(0,self.M)]=1
```

首先生成一个矩阵，行表示y的种类，纵坐标表示y的个数，通过直接进行赋值得到一个稀疏矩阵，然后直接和刚刚得到的矩阵进行计算就可以了


```python
def train(self,maxiter=200,disp = False):
		#res,f,d=sp.optimize.fmin_l_bfgs_b(self.LRcost,self.theta,disp=1)
		x0=np.random.rand(self.classNum,self.N).reshape(self.classNum*self.N)/10
		res=sp.optimize.minimize(self.LRcost,x0, method='L-BFGS-B',jac=True,options={'disp': disp,'maxiter': maxiter})
		self.theta=res.x   #获得参变量系数值
		pass
```

这里是用optimize的BFGS来进行计算的。

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML">
</script>