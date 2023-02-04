## Non-parametric Bayesian Models

### Introduction

我们将要从以下方面谈论无参贝叶斯：

+ 回归、分类：高斯过程
+ 聚类：狄利克雷过程

对于贝叶斯模型：

在训练时，我们只是估计$f$的后验估计：
$$
p(f\mid D) = \frac{p(D\mid f)p(f)}{p(D)}
$$
在预测时，我们不再需要点估计。相反，我们考虑所有可能的模型并且取期望：
$$
p(y\mid x,D) = \int_f p(y\mid f,x) p(f\mid D)df
$$
为什么要选用贝叶斯呢？

对于无限可交换：
$$
\forall n,\forall \sigma, \quad p(x_1,\cdots,x_n) = p(x_{\sigma(1)},\cdots,x_{\sigma(n)})
$$
`De Finetti's`定理：如果$(x_1,x_2,\cdots)$为无限可交换的，则存在随机变量$f,\forall n$：
$$
p(x_1,\cdots,x_n) = \int_f\prod_{i=1}^n p(x_i\mid f)p(f)df
$$
但是怎么定义先验$f$呢？

### Gaussian Process

我们先复习一下高斯分布：
$$
x \sim \mathcal{N}(\mu,\Sigma)
$$
其中
$$
p(x) = \frac{1}{\sqrt{(2\pi)^n|\Sigma|}}\exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)
$$
其中
$$
\mathbb{E}[x] = \mu
$$

$$
\mathbb{E}[(x_i-\mu_i)(x_j-\mu_j)] = \Sigma_{ij}
$$

假设
$$
\begin{bmatrix}x_1\\x_2\end{bmatrix}\sim \mathcal{N}\left(\begin{bmatrix}\mu_1\\\mu_2\end{bmatrix}, \begin{bmatrix}\Sigma_{11}&\Sigma_{12}\\\Sigma_{21}&\Sigma_{22}\end{bmatrix}\right)
$$
则
$$
x_1 \sim \mathcal{N}(\mu_1,\Sigma_{11})
$$
条件分布：
$$
\begin{aligned}
p(x_1\mid x_2) &= \mathcal{N}(x_1\mid \mu_{1\mid 2},\Sigma_{1\mid 2})\\
\mu_{1\mid 2} &= \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2)\\
\Sigma_{1\mid 2} &= \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
\end{aligned}
$$
证明：

首先我们先对协方差矩阵进行对角化，方便对其取逆：
$$
\left[\begin{array}{cc}
\Sigma_{11} / \Sigma_{22} & 0 \\
0 & \Sigma_{22}
\end{array}\right]=\left[\begin{array}{cc}
I_1 & -\Sigma_{12} \Sigma_{22}^{-1} \\
0 & l_2
\end{array}\right]\left[\begin{array}{cc}
\Sigma_{11} & \Sigma_{12} \\
\Sigma_{21} & \Sigma_{22}
\end{array}\right]\left[\begin{array}{cc}
I_1 & 0 \\
-\Sigma_{22}^{-1} \Sigma_{21} & I_2
\end{array}\right]
$$


