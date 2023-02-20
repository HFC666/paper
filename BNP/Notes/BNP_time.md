# A Bayesian Nonparametric Approach for Time Series Clustering

## The model

### Sampling model

令 $\mathrm{y}_i = \{y_{it}:t=1,2,\cdots,T\}, i=1,\cdots,n$为$n$个时间序列的集合，每一个都是$T$时间长度。其中一个用来分析时间序列的强大的贝叶斯模型是**动态线性模型**：
$$
\begin{aligned}
    y_{it} &= F_{it}\theta_{it} + \epsilon_{it}\\
    \theta_{it} &= \rho\theta_{i,t-1} + \nu_{it}
\end{aligned}
$$
其中$\epsilon_{it}\sim \mathrm{N}(0,\sigma_{\epsilon i}^2)$并且$\nu_{it}\sim\mathrm{N}(0,\sigma_\theta^2)$。

我们将注意力集中在第二个公式上面并且去除掉索引$i$：$\theta_t = \rho\theta_{t-1}+\nu_t$。