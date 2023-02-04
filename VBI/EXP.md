---
title: 指数分布
categories: 论文阅读
tags: [机器学习, 统计]
update: 2022-7-9
state: 完成
---

## Exponential Family Distribution

![](D:\joplin\2022\2022\paper\VBI\EX.png)

### 背景

指数分布族是可以写为如下形式的分布：
$$
P(x\mid \eta) = h(x)\exp\left(\eta^T\phi(x)-A(\eta)\right)
$$
其中$\eta$为参数向量，$x\in \mathbb{R}^p$，$A(\eta)$为对数配分函数(log partition function)。

下面我们解释一下**配分函数**，配分函数可以理解为**归一化因子**，例如我们在无向图模型中经常用到的：
$$
P(X\mid \theta) = \frac{1}{Z}\hat{P}(X\mid \theta)
$$
其中$\hat{P}(X\mid \theta)$是我们构造出来的分布，但是概率分布必须满足和为$1$，所以我们在前面添加归一化因子使得：
$$
Z = \int_x \hat{P}(x\mid \theta)dx
$$
可以看出$Z$与$X$无关，那为什么$A(\eta)$称为对数配分函数呢？这是因为：
$$
\begin{aligned}
P(x\mid \eta) &= h(x)\exp\left(\eta^T\phi(x)-A(\eta)\right)\\
&= \frac{1}{\exp(A(\eta))}h(x)\exp(\eta^T\phi(x))
\end{aligned}
$$
所以$Z = \exp(A(\eta))\Rightarrow A(\eta)=\ln Z$，所以其被称为对数配分函数。

其中$\phi(x)$为**充分统计量**。充分统计量指的是能够包含样本中所有信息的统计量。

如对于数据$x_1,\cdots,x_N$，我们假设其服从于高斯分布，那么其充分统计量就为：
$$
\phi(x) = \begin{bmatrix}\sum_{i=1}^Nx_i\\\sum_{i=1}^Nx_i^2\end{bmatrix}
$$
因为有了这两个统计量我们就可以求出其**均值**和**方差**。

在贝叶斯推断中我们常常遇到这样的问题：
$$
P(Z\mid X) = \frac{P(X\mid Z)P(Z)}{\int_ZP(X\mid Z)P(Z)dZ}
$$
有时候积分很难算出，即使积分算出了，$P(Z\mid X)$的形式可能很复杂，我们无法求解其期望和方差，这时候我们可以采用采样的方法(MCMC)或者通过变分推断来寻找接近$P(Z\mid X)$的概率分布$Q(X)$。

但是指数族分布可以采用共轭的性质。

指数族分布与广义线性模型，广义线性模型的重要组成部分为：

1. 线性组合，如$w^Tx$
2. link function：为激活函数的逆函数
3. 指数族分布：$y\mid x\sim$指数族分布

概率图模型中非常重要的一组模型为无向图RBF，与指数族分布具有非常重要的关系。

另外当分布为指数族分布时，变分推断可见极大地简化。

### 高斯分布的指数族形式

高斯分布的形式为：
$$
\begin{aligned}
P(x\mid\theta) &= \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{1}{2\sigma^2}(x-\mu)^2\right)\quad \theta = (\mu,\sigma^2)\\
&=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{1}{2\sigma^2}(x^2-2\mu x+\mu^2)\right)\\
&=\exp\left(\log(2\pi\sigma^2)^{-\frac{1}{2}}\right)\exp\left(-\frac{1}{2\sigma^2}\begin{pmatrix}-2\mu&1\end{pmatrix}\begin{pmatrix}x\\x^2\end{pmatrix}-\frac{\mu^2}{2\sigma^2}\right)\\
&=\exp\left(\underbrace{\begin{pmatrix}\frac{\mu}{\sigma^2}&-\frac{1}{2\sigma^2}\end{pmatrix}}_{\eta^T}\cdot\underbrace{\begin{pmatrix}x\\x^2\end{pmatrix}}_{\phi(x)}-\underbrace{\left(\frac{\mu^2}{2\sigma^2}+\frac{1}{2}\log2\pi\sigma^2\right)}_{A(\eta)}\right)
\end{aligned}
$$
其中我们令$\eta_1=\frac{\mu}{\sigma^2},\eta_2=-\frac{1}{2\sigma^2}$。则$\sigma^2=-\frac{1}{2\eta_2},\mu=-\frac{\eta_1}{2\eta_2}$。代入$A(\eta)$，得：
$$
A(\eta) = -\frac{\eta_1^2}{4\eta_2}+\frac{1}{2}\log\left(-\frac{\pi}{\eta_2}\right)
$$

### 对数配分函数与充分统计量的关系



我们之前提到过：
$$
\exp(A(\eta)) = \int_x h(x)\exp(\eta^T\phi(x))dx
$$
两边同时对$\eta$求导，得：
$$
\begin{aligned}
\exp(A(\eta))\cdot A^\prime(\eta) &= \frac{\partial}{\partial \eta}(\int h(x)\exp(\eta^T\phi(x))dx)\\
&= \int_xh(x)\exp(\eta^T\phi(x))\phi(x)dx
\end{aligned}
$$
两边同除以$A^\prime(\eta)$，得
$$
\begin{aligned}
A^\prime(\eta) &= \frac{\int_xh(x)\exp(\eta^T\phi(x))\phi(x)dx}{\exp(A(\eta))}\\
&=\int_x \underbrace{h(x)\exp(\eta^T\phi(x)-A(\eta))}_{P(x\mid\eta)}\phi(x)dx\\
&= \mathbb{E}_{P(x\mid\eta)}[\phi(x)]
\end{aligned}
$$
所以$A^\prime(\eta) = \mathbb{E}_{P(x\mid \eta)}[\phi(x)]$。

同样地，我们也可以研究一下二阶导，对式子
$$
A^\prime(\eta) =\int_x \underbrace{h(x)\exp(\eta^T\phi(x)-A(\eta))}_{P(x\mid\eta)}\phi(x)dx
$$
两边同时求导得：
$$
\begin{aligned}
A^{\prime\prime}(\eta) &= \int_x \underbrace{h(x)\exp(\eta^T\phi(x)-A(\eta))}_{P(x\mid\eta)}(\phi(x)-A^{\prime}(\eta))\phi(x)dx\\
&= \int_xP(x\mid\eta)(\phi(x)-\mathbb{E}_{P(x\mid\eta)}[\phi(x)])\phi(x)dx\\
&= \int_x P(x\mid \eta)\phi(x)^2dx - \mathbb{E}_{P(x\mid\eta)}[\phi(x)])\int_xP(x\mid \eta)\phi(x)dx\\
&= \mathbb{E}_{P(x\mid\eta)}[\phi(x)^2] - \left(\mathbb{E}_{P(x\mid \eta)}[\phi(x)]\right)^2\\
&=\operatorname{Var}[\phi(x)]
\end{aligned}
$$

### 极大似然估计与充分统计量

假设我们的数据为：$D=\{x_1,x_2,\cdots,x_N\}$，所以我们有：
$$
\begin{aligned}
\eta_{\text{mle}} &= \arg\max\log P(D\mid \eta)\\
&=\arg\max \log\prod_{i=1}^N P(x_i\mid \eta)\\
&=\arg\max \sum_{i=1}^N\log P(x_i\mid \eta)\\
&=\arg\max\sum_{i=1}^N\log\left[h(x_i)\exp(\eta^T\phi(x_i)-A(\eta))\right]\\
&= \arg\max\sum_{i=1}^N\left[\log h(x_i)+\eta^T\phi(x_i)-A(\eta)\right]\\
&= \arg\max\sum_{i=1}^N(\eta^T\phi(x_i)-A(\eta))
\end{aligned}
$$
我们对其求导，得：
$$
\begin{aligned}
\frac{\partial}{\partial \eta}\left(\sum_{i=1}^N\eta^T\phi(x_i)-A(\eta)\right)&=\sum_{i=1}^N\frac{\partial}{\partial \eta}(\eta^T\phi(x_i)-A(\eta))\\
&=\sum_{i=1}^N\phi(x_i)-A^{\prime}(\eta)\\
&=\sum_{i=1}^N\phi(x_i)-NA^{\prime}(\eta)
\end{aligned}
$$
令导数等于$0$，得
$$
A^{\prime}(\eta_{\text{mle}}) = \frac{1}{N}\sum_{i=1}^N\phi(x_i)
$$
这样我们就可以求出$\eta_{\text{mle}}$，可以看出$\eta_{\text{mle}}$仅与$\phi(x)$有关，即确定了$\phi(x)$即确定了$\eta_{\text{mle}}$，即确定了分布，更加验证了$\phi(x)$充分统计量的结论。

### 最大熵角度

假设一个事件发生的概率为$p$，其信息量为$-\log p$。熵的概念就是信息量关于分布本身的期望：
$$
\begin{aligned}
\mathbb{E}_{p(x)}[-\log p] &= -\int_x p(x)\log p(x)dx\\
&= -\sum_x p(x)\log p(x)
\end{aligned}
$$
最大熵的思想通俗来说就是等可能的，当我们对一个事件一无所知时，我们一般假设其是等可能的。下面看一个例子：

我们用$H[P]$来表示熵：
$$
H[p] = -\sum_x p(x)\log p(x)
$$
我们假设$x$是离散的，$x$可以取值的个数为$K$，概率分别对应于$p_1,\cdots,p_K$，并且$\sum_{i=1}^K p_i=1$。所以其熵为：
$$
H[p] = -\sum_{i=1}^K p_i\log (p_i)
$$
我们令其最大，即变为了优化问题：
$$
\begin{aligned}
&\min \sum_{i=1}^K p_i\log p_i\\
&\text{s.t. }\sum_{i=1}^K p_i=1
\end{aligned}
$$
我们可以直接用拉格朗日乘子法进行求解，定义：
$$
\mathcal{L}(p,\lambda) = \sum_{i=1}^K p_i\log(p_i)+\lambda(1-\sum_{i=1}^Kp_i)
$$
对$p_i$求导，得：
$$
\frac{\partial \mathcal{L}(p,\lambda)}{\partial p_i} = \log(p_i)+p_i\cdot\frac{1}{p_i}-\lambda
$$
所以：
$$
p_i = \exp(\lambda-1)
$$
对于每个$p_i$都是如此，所以$p_1=p_2=\cdots=p_K=\frac{1}{K}$，所以其为等可能的。



最大熵原理：在满足既定事实的前提下，具有最大熵的分布即为我们想要的分布。在机器学习中，我们的既定事实即为数据，假设我们的数据为$D=\{x_1,\cdots,x_N\}$。

在这里我们引入经验分布的概念，其是对已知样本的描述，其定义为：
$$
\hat{p}(X=x) = \frac{\text{count}(x)}{N}
$$
因为分布$\hat{p}$我们已经求出来了，所以对于$x$的任意函数$f(x)$向量，我们也能求其期望：
$$
\mathbb{E}_{\hat{p}}[f(x)] = \Delta(\text{已知})
$$
这个就是我们的**已知约束**。

下面我们求满足上述约束的最大熵的分布，这就变成了优化问题：
$$
\begin{aligned}
&\min \sum_x p(x)\log p(x)\\
&\text{s.t. } \sum_x p(x)=1\\
&\quad\quad \mathbb{E}_{\hat{p}} [f(x) ]= \mathbb{E}_p[f(x)] = \Delta
\end{aligned}
$$
定义拉格朗日乘子：
$$
\mathcal{L}(p,\lambda_0,\lambda) = \sum_x p(x)\log p(x) + \lambda_0(1-\sum_x p(x))+\lambda^T(\Delta-\mathbb{E}_p[f(x)])
$$
对$p(x)$求导得：
$$
\frac{\mathcal{L}}{\partial p(x)} =(\log p(x)+1)-\lambda_0-\lambda^T f(x)
$$
令其等于$0$，得
$$
\log p(x) = \lambda^T f(x) + \lambda_0-1
$$
所以
$$
p(x) = \exp\left(\lambda^T f(x) - (1-\lambda_0)\right)
$$
为指数族分布。



