## Normalizing Flows

我们在应用变分推断时，常常规定变分分布为特定类型的分布，这有时会不符合实际情况。现在我们引入一种Normalizing Flows，可以使用一系列可逆映射来近似任意复杂的概率分布。



### Finite Flows

概率变换的基本规则考虑一个可逆的、平滑的映射$f:\mathbb{R}^d\rightarrow \mathbb{R}^d$，逆为$f^{-1}=g$，则$g\circ f(z)=z$。如果我们使用这个映射来变换一个概率密度为$q(z)$的随机变量$z$，则得到的随机变量$z^\prime = f(z)$的概率分布为：
$$
q(z^\prime) = q(z)\left|\det\frac{\partial f^{-1}}{\partial z^\prime}\right| = q(z) \left|\det \frac{\partial f}{\partial z}\right|^{-1}
$$
如果我们应用多次概率变换：
$$
\begin{aligned}
z_k &= f_K\circ\cdots\circ f_2\circ f_1(z_0)\\
\ln q_K(z_K) &= \ln q_0(z_0) - \sum_{k=1}^K\ln \det\left|\frac{\partial f_k}{\partial z_k}\right|
\end{aligned}
$$
任何期望$\mathbb{E}_{q_K}[h(z)]$可以被写为$q_0$下的期望：
$$
\mathbb{E}_{q_K}[h(z)] = \mathbb{E}_{q_0}[h(f_K\circ f_{K-1}\circ\cdots\circ f_1(z_0))]
$$
则我们不需要计算对数行列式项，当$h(z)$不依赖于$q_K$。

有了合适的变换$f_K$后，我们用简单的可分解的分布如独立高斯，应用不同长度的normalizing flow来提高分布的复杂性。

### Infinitesimal Flows

当normalizing flow是无限长的，在这种情况下我们得到一个infinitesimal flow，我们将其描述为偏微分方程
$$
\frac{\partial}{\partial t}q_t(z) = \mathcal{T}_t[q_t(z)]
$$
