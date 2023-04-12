## Shifting means and hidden Markov models

### Shifting means models

我们介绍两种SMM模型。

SMM1可以被表示为：
$$
x_t = m_t + \epsilon_t,\quad m_t = m_{t-1} + z_{t-1}\cdot \delta_t
$$

1. $\epsilon_1,\epsilon_2,\cdots$为独立同分布的随机变量，服从分布$\mathcal{N}(0,\sigma_\epsilon)$
2. $z_1,z_2,\cdots$为独立同分布的随机变量，取值$0,1$，概率分布为$\eta = \Pr(z_t=1),1-\eta=\Pr(z_t=0)$
3. $\delta_1,\delta_2,\cdots$为独立同分布的随机变量，服从分布$\mathcal{N}(0,\sigma_\mu)$。

其参数为：
$$
\theta = (\eta,\sigma_\epsilon,\sigma_\mu)
$$
SMM2可以表示为：
$$
x_t = m_t + \epsilon_t\quad m_t = (1-z_{t-1})\cdot m_{t-1}+z_{t-1}\cdot (\mu + \delta_t)
$$
其参数为：
$$
\theta = (\eta,\mu,\sigma_\epsilon,\sigma_\mu)
$$
我们可以将SMM看作是一个隐马尔可夫模型，其状态为$(m_t,z_t)$，观测为$x_t$，但是因为$m_t\in \mathbb{R}$，所以状态是不可数的，这就导致估计参数的困难。

### Hidden Markov models which emulate SMM’s

我们可以用有限维来处理SMM的无限维。标准方法是将$\mathbb{R}$用有限集合$\mathrm{S} = \{\mu_1,\mu_2,\cdots,\mu_K\}$代替。

我们引入一个马尔可夫随机过程$s_1,s_2,\cdots$，取值为$\mathrm{S} = \{1,2,\cdots,K\}$。对于每一个状态都有一个参数$\mu_k(k=1,2,\cdots,K)$。我们假设$x_t$的条件概率，给定$s_t=k$，为$\mathcal{N}(\mu_k, \sigma_\epsilon)$。换句话说，发射函数$\mathrm{f}(x) = [f_k(x)]_{k=1}^K$定义为：
$$
f_k(x) = \frac{1}{\sqrt{2\pi}\sigma_\epsilon}\exp\left\{-\frac{1}{2}\cdot\left(\frac{x-\mu_k}{\sigma_\epsilon}\right)\right\}
$$
还差概率转移矩阵$\mathrm{P}$。我们提供4种不同的选择。
$$
P_{jk} = \begin{cases}
(1-\eta) + \eta\cdot g_{jk}&\text{ if }j=k\\
\eta\cdot g_{jk}&\text{ if }j\neq k
\end{cases}
$$
唯一不同的是$g_{jk}$。

HMM1，在HMM1中，我们定义$g_{jk}$为：
$$
\begin{aligned}
& g_{j k}=c_j \cdot e^{-\left(\mu_k-\mu_j\right)^2 / 2 \sigma_\mu^2} \\
& c_j=\left(\sum_{k=1}^K e^{-\left(\mu_k-\mu_j\right)^2 / 2 \sigma_\mu^2}\right)^{-1}(j, k=1,2, \ldots, K)
\end{aligned}
$$
HMM2：
$$
\begin{aligned}
& g_{j 1}=\frac{1}{\sqrt{2 \pi} \sigma_\mu} \int_{-\infty}^{\frac{\mu_2+\mu_1}{2}} e^{-\left(z-\mu_j\right)^2 / 2 \sigma_\mu^2} d z \\
& g_{j k}=\frac{1}{\sqrt{2 \pi} \sigma_\mu} \int_{\frac{\mu_{k+1}+\mu_{k-1}}{2}}^{\frac{\mu_k+\mu_k}{2}} e^{-\left(z-\mu_j\right)^2 / 2 \sigma_\mu^2} d z(k=2,3, \ldots, K-1) \\
& g_{j K}=\frac{1}{\sqrt{2 \pi} \sigma_\mu} \int_{\frac{\mu_K+\mu_{K-1}}{2}}^{\infty} e^{-\left(z-\mu_j\right)^2 / 2 \sigma_\mu^2} d z
\end{aligned}
$$
HMM3：
$$
\begin{aligned}
& g_{j k}=c_j \cdot e^{-\left(\mu_k-\mu\right)^2 / 2 \sigma_\mu^2} \\
& c_j=\left(\sum_{k=1}^K e^{-\left(\mu_k-\mu\right)^2 / 2 \sigma_\mu^2}\right)^{-1}(j, k=1,2, \ldots, K)
\end{aligned}
$$
HMM4：
$$
\begin{aligned}
& g_{j 1}=\frac{1}{\sqrt{2 \pi} \sigma_\mu} \int_{-\infty}^{\frac{\mu_2+\mu_1}{2}} e^{-(z-\mu)^2 / 2 \sigma_\mu^2} d z \\
& g_{j k}=\frac{1}{\sqrt{2 \pi} \sigma_\mu} \int_{\frac{\mu_k+\mu_{k-1}}{2}}^{\frac{\mu_{k+1}+\mu_k}{2}} e^{-(z-\mu)^2 / 2 \sigma_\mu^2} d z(k=2,3, \ldots, K-1) \\
& g_{j K}=\frac{1}{\sqrt{2 \pi} \sigma_\mu} \int_{\frac{\mu_K+\mu_{K-1}}{2}}^{\infty} e^{-(z-\mu)^2 / 2 \sigma_\mu^2} d z
\end{aligned}
$$
