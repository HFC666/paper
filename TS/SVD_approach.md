## 时间序列分割

我们定义一个多元时间序列$T = \{x_k = [x_{1,k},x_{2,k},\cdots,x_{m,k}]\mid 1\le k \le N\}$为含有$n$个采样点的$m$维观测，我们表示为$t_1,\cdots,t_N$。因此，$T$的一个片段定义为一系列连续的时间点$S(a,b)=a\le k\le b,x_a,x_{a+1},\cdots,x_b$。将时间序列$T$分割为$c$个不重叠的时间区间可以被表示为$S_T^c = \{S_e(a_e,b_e)\mid 1\le e\le c\}$，其中$a_1 = 1,b_c=N$并且$a_e = b_{e-1}+1$。

由于我们的方法旨在检测多个变量之间相关结构的变化，因此分割的cost函数基于分割矩阵的奇异值分解，其中每一行都是一个观察值，每一列都是一个变量。在应用 SVD 之前，需要对观察到的变量进行居中和缩放，以使它们具有可比性。

SVD分解：
$$
X \approx U\Sigma V^T
$$
因为我们关心的是观测向量之间的关系，因此投影矩阵$P$只考虑矩阵$V$。我们用$Q$测度作为同质性的衡量指标。我们的cost函数定义为：

$$
\begin{aligned}
P_{m,n} &= \sum_{k=1}^p X_{m,n}V_{n,k}V_{k,n}^T\\
Q&= \frac{1}{m\cdot n}\sum_{i=1}^m\sum_{j=1}^n P_{i,j}^2\\
\text{cost}_Q(S_e(a_e,b_e)) &= \frac{1}{b_e-e_e+1}\sum_{k=a_e}^{b_e}Q_{e,k}
\end{aligned}
$$

