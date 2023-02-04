## Gaussian Process State-Space Models

我们考虑离散时间非线性状态空间模型：
$$
\begin{aligned}
\mathbf{x}_{t+1} &= f(\mathbf{x}_t) + \mathrm{v}_t\\
\mathbf{y}_t &= g(\mathbf{x_t}) + \mathbf{e}_t 
\end{aligned}
$$
一个传统的做法是通过将$f,g$限制为一族参数模型来学习其参数，但是有时候选择什么样的参数模型很困难。高斯过程可以表示任意复杂的函数，并提供了一种直接的方法来指定对那些未知函数的假设，因此我们可以为$f,g$设置高斯过程的先验。但是由于高斯过程惊异的灵活性导致了两个未知函数的后验之间存在严重的不可识别性和强相关性。在本文的剩余部分，我们将关注一个具有GP先验的状态转移函数和参数似然的模型。

状态转移函数为高斯过程先验的概率状态-空间模型表示为：
$$
\begin{aligned}
f(x) &\sim \mathcal{GP}(m_f(x), k_f(x,x^\prime))\\
x_t\mid f_t &\sim \mathcal{N}(x_t\mid f_t,Q)\\
x_0 &\sim p(x_0)\\
y_t\mid x_t &\sim p(y_t\mid x_t,\theta_y)
\end{aligned}
$$
其中$f_t \triangleq f(x_{t-1})$。我们将参数分组：$\mathbf{\theta} = \triangleq \{\mathbf{\theta_f}, \mathbf{\theta_y},\mathbf{Q}\}$。我们的联合概率分布可以表示为：
$$
p(\mathbf{y,x,f}) = p(\mathbf{x_0})\prod_{t=1}^T p(\mathrm{y_t}\mid \mathrm{x_t}) p(\mathrm{x}_t\mid \mathrm{f_t})p(\mathrm{f_t}\mid \mathrm{f_{1:t-1}}, \mathrm{x_{0:t-1}})
$$

我们规定$\mathrm{f_{1:0}} = \varnothing$。我们可以得到：
$$
p(\mathrm{f_t}\mid \mathrm{f_{1:t-1}},\mathrm{x_{0:t-1}}) = \mathcal{N}(m_f(\mathrm{x}_{t-1}) + \mathbf{K}_{t-1,0:t-2}\mathbf{K}_{0:t-2,0:t-2}^{-1}(\mathrm{f_{1:t-1}}-m_f(\mathrm{x_{0:t-2}})), \mathbf{K}_{t-1,t-1}-\mathbf{K}_{t-1,0:t-2}\mathbf{K}_{0:t-2,0:t-2}^{-1}\mathbf{K}_{t-1,0:t-2}^\top)
$$
其中：$\mathrm{K}_{t-1,0:t-2} = [k_f(x_{t-1},x_0)\cdots k_f(x_{t-1},x_{t-2})]$。当$t=1$时，$p(f_1\mid x_0) = \mathcal{N}(m_f(x_0), k_f(x_0,x_0))$。



### Variational Inference in GP-SSMs

#### Augmenting the Model with Inducing Variables

我们利用$M$个诱导点(inducing points)$\mathbf{u}\triangleq \{\mathbf{u_i}\}_{i=1}^M$对模型进行扩充，我们的联合概率密度函数变为：
$$
p(\mathbf{y,x,f,u}) = p(\mathrm{x,f\mid u}) p(\mathbf{u}) \prod_{t=1}^T p(\mathbf{y}_t\mid \mathbf{x}_t)
$$
其中
$$
\begin{aligned}
p(\mathbf{u}) &=\mathcal{N}\left(\mathbf{u} \mid \mathbf{0}, \mathbf{K}_{\mathbf{u}, \mathbf{u}}\right) \\
p(\mathbf{x}, \mathbf{f} \mid \mathbf{u}) &=p\left(\mathbf{x}_0\right) \prod_{t=1}^T p\left(\mathbf{f}_t \mid \mathbf{f}_{1: t-1}, \mathbf{x}_{0: t-1}, \mathbf{u}\right) p\left(\mathbf{x}_t \mid \mathbf{f}_t\right) \\
\prod_{t=1}^T p\left(\mathbf{f}_t \mid \mathbf{f}_{1: t-1}, \mathbf{x}_{0: t-1}, \mathbf{u}\right) &=\mathcal{N}\left(\mathbf{f}_{1: T} \mid \mathbf{K}_{0: T-1, \mathbf{u}} \mathbf{K}_{\mathbf{u}, \mathbf{u}}^{-1} \mathbf{u}, \mathbf{K}_{0: T-1}-\mathbf{K}_{0: T-1, \mathbf{u}} \mathbf{K}_{\mathbf{u}, \mathbf{u}}^{-1} \mathbf{K}_{0: T-1, \mathbf{u}}^{\top}\right) .
\end{aligned}
$$
最后一个式子的左边连乘得到的结果为：$p(\mathbf{f}_{1:T}\mid \mathbf{x}_{0:T-1},\mathbf{u})$。

> 稀疏高斯过程的思想是选取几个点$\mathbf{u}$来表示高斯过程，将高斯过程表示完毕后，我们的数据就是独立同分布的，每一个点都相当于一个新加入的点，可以利用条件高斯分布的性质来得到其分布：
> $$
> p(\mathbf{f}\mid \mathbf{x},\mathbf{u}) = \mathcal{N}(\mathbf{K_{x,u}}^\top \mathbf{K_u}^{-1}\mathbf{u}, \mathbf{K_{x}}-\mathbf{K_{x,u}}\mathbf{K_{u,u}}^\top\mathbf{K_{x,u}})
> $$
> 最后将所有点的概率密度函数的均值和协方差堆叠得到所有点的联合概率分布函数。

#### Evidence Lower Bound of an Augmented GP-SSM

利用琴声不等式得到下界，对下界进行优化：
$$
\log p(\mathbf{y} \mid \boldsymbol{\theta}) \geq \int_{\mathbf{x}, \mathbf{f}, \mathbf{u}} q(\mathbf{x}, \mathbf{f}, \mathbf{u}) \log \frac{p(\mathbf{u}) p\left(\mathbf{x}_0\right) \prod_{t=1}^T p\left(\mathbf{f}_t \mid \mathbf{f}_{1: t-1}, \mathbf{x}_{0: t-1}, \mathbf{u}\right) p\left(\mathbf{y}_t \mid \mathbf{x}_t\right) p\left(\mathbf{x}_t \mid \mathbf{f}_t\right)}{q(\mathbf{x}, \mathbf{f}, \mathbf{u})}
$$
我们采用如下所示的变分分布：
$$
q(\mathbf{x,f,u}) = q(\mathbf{u})q(\mathbf{x})\prod_{t=1}^T p(\mathbf{f}_t\mid \mathbf{f}_{1:t-1},\mathbf{x}_{0:t-1},\mathbf{u})
$$
整理以下模型我们可以得到：
$$
\begin{aligned}
\mathcal{L}(q(\mathbf{u}), q(\mathbf{x}), \boldsymbol{\theta}) &=-\mathrm{KL}(q(\mathbf{u}) \| p(\mathbf{u}))+\mathcal{H}(q(\mathbf{x}))+\int_{\mathbf{x}} q(\mathbf{x}) \log p\left(\mathbf{x}_0\right) \\
&+\sum_{t=1}^T\{\int_{\mathbf{x}, \mathbf{u}} q(\mathbf{x}) q(\mathbf{u}) \underbrace{\int_{\mathbf{f}_t} p\left(\mathbf{f}_t \mid \mathbf{x}_{t-1}, \mathbf{u}\right) \log p\left(\mathbf{x}_t \mid \mathbf{f}_t\right)}_{\Phi\left(\mathbf{x}_t, \mathbf{x}_{t-1}, \mathbf{u}\right)}+\int_{\mathbf{x}} q(\mathbf{x}) \log p\left(\mathbf{y}_t \mid \mathbf{x}_t\right)\}
\end{aligned}
$$

其中KL表示KL散度并且$\mathcal{H}$表示熵。其中与$\mathbf{f}_t$相关的积分存在解析解：$\Phi(\mathbf{x_t,x_{t-1},u}) = -\frac{1}{2}\text{tr}(\mathbf{Q}^{-1}\mathbf{B}_{t-1})+\log \mathcal{N}(\mathbf{x}_{t}\mid \mathbf{A_{t-1}u,Q})$其中$\mathbf{A_{t-1}} = \mathbf{K_{t-1,u}K_{u,u}^{-1}}$并且$\mathbf{B}_{t-1} = \mathbf{K_{t-1,t-1}}-\mathbf{K_{t-1,u}K_{u,u}^{-1}K_{u,t-1}}$。

#### Optimal Variational Distribution for $\mathbf{u}$

