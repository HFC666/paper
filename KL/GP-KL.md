## An Introduction to Gaussian Processes for the Kalman Filter Expert

### The Kalman Filter

假设我们的观测变量为$y_k$，我们有：
$$
y_k = H_k f_k + \epsilon_k
$$
$H_k$为线性观测模型，$\epsilon_k$服从均值为零协方差为$Q$的多元高斯分布。KF决定基于观测$y_j(j\le k)$的$f_k$的均值$\bar{f}_{k\mid k}$和协方差$P_{k\mid k}$。令$\bar{f}_{k\mid k-1}$和$P_{k\mid k-1}$分别表示给定观测$y_j(j<k)$后的$f_k$分布的均值和协方差。则KF融合的公式变为：
$$
\begin{aligned}
\bar{f}_{k\mid k} &= \bar{f}_{k\mid k-1} + \Gamma_k[y_k-H_k\bar{f}_{k\mid k-1}]\\
P_{k\mid k} &= [I-\Gamma_kH_k]P_{k\mid k-1}
\end{aligned}
$$
其中
$$
\Gamma_k = P_{k\mid k-1}H_k^T[H_kP_{k\mid k-1}H_k^T+R]^{-1}
$$
为卡尔曼增益。



现在我们把注意力转移到函数的推断上。令$f(x)$为向量空间$x$上的非线性函数。其中$x$为输入向量并且$f(x)$为输出标量。在第$k$次迭代我们可能会得到这个函数在$n$个输入变量$X=\{x_1,\cdots,x_n\}$处的观测$Y_k$。观测可能有噪声但是我们假设：
$$
Y_k = f(X)+\epsilon_k
$$
其中$\epsilon_k \sim \mathcal{N}(0, R),R=\sigma^2 I$。假设我们想要去推断在$m$个输入$X_\star = \{x_1^\star,\cdots,x_m^\star\}$处的函数值：
$$
f(X_\star) = (f(x_1^\star),\cdots,f(x_m^\star))
$$
为了清晰我们定义$X^\prime\triangleq X_\star \cup X$并且指示矩阵$H^\star$和$H_k$使得$X_\star = H^\star X^\prime$和$X = H_kX^\prime$，则：
$$
Y_k = H_kf(X^\prime) + \epsilon_k
$$
现在，假设我们有$f(X^\prime)$的高斯先验分布，均值和方差分别为$\bar{f}_{\mid 0}(X^\prime)$和$P_{\mid 0}(X^\prime,X^\prime)$。则我们可以对状态向量$f(X^\prime)$应用KF融合得到：
$$
\begin{aligned}
\bar{f}_{\mid k}\left(X^{\prime}\right) &=\bar{f}_{\mid k-1}\left(X^{\prime}\right)+\Gamma_k\left[Y_k-H_k \bar{f}_{\mid k-1}\left(X^{\prime}\right)\right], \\
P_{\mid k}\left(X^{\prime}, X^{\prime}\right) &=\left[I-\Gamma_k H_k\right] P_{\mid k-1}\left(X^{\prime}, X^{\prime}\right)
\end{aligned}
$$
其中
$$
\Gamma_k = P_{\mid k-1}(X^\prime,X^\prime)H_k^T[H_k P_{\mid k-1}(X^\prime,X^\prime)H_k^T+\sigma^2I]^{-1}
$$
我们很容易得到$\bar{f}_{\mid k-1}(X) = H_k \bar{f}_{\mid k-1}(X^\prime)$和$P_{\mid k-1}(X,X) = H_kP_{\mid k-1}(X^\prime,X^\prime)H_k^T$，我们只对$X^\prime$进行KF融合：
$$
\begin{aligned}
\bar{f}_{\mid k}(X_\star) &= \bar{f}_{\mid k-1}(X_\star) + P_{\mid k-1}(X_\star,X)[P_{\mid k-1}(X,X) + \sigma^2I]^{-1}[Y_k - \bar{f}_{\mid k-1}(X)]\\
P_{\mid k}(X_\star,X_\star) &= P_{\mid k-1}(X_\star,X_\star)-P_{\mid k-1}(X_\star,X)[P_{\mid k-1}(X,X)+\sigma^2 I]^{-1}P_{\mid k-1}(X,X_\star)
\end{aligned}
$$

### Gaussian Processes

假设我们的训练数据是从有噪声的过程中产生的：
$$
y_i = f(x_i) + \epsilon_i, \epsilon_i \sim \mathcal{N}(0,\sigma^2)
$$
GP估计在任意预测点$X_\star = \{x_{\star1},\cdots,x_{\star m}\}$出的函数值$f$为：
$$
\begin{aligned}
\bar{f}_\star &= m(X_\star) + K(X_\star,X)[K(X,X)+\sigma^2I]^{-1}\times (Y-m(X))\\
\text{Cov}(f_\star) &= K(X_\star,X_\star) - K(X_\star,X)\times [K(X,X)+\sigma^2 I]^{-1}K(X_\star,X)^T
\end{aligned}
$$

### The Augmented Kalman Smoother (AKS)

将多元高斯随机变量$f$分割为：
$$
f = \begin{pmatrix}f_A\\f_B \end{pmatrix}
$$
定义先验均值：
$$
\bar{f} = \begin{pmatrix} \bar{f}_A \\ \bar{f}_B \end{pmatrix}
$$
先验协方差：
$$
K = \begin{pmatrix}K_{A,A} &K_{A,B}\\K_{A,B}^T &K_{B,B} \end{pmatrix}
$$
令$\bar{f}_{B\mid z_B}$和$P_{B\mid z_B}$分别表示$f_B$的后验均值和协方差。如果$z_B$在给定$f_B$的条件下与$f_A$是条件独立的，则$f$的后验均值$\bar{f}_{\mid z_B}$和协方差$P_{\mid z_B}$为：
$$
\bar{f}_{\mid z_B} = \begin{pmatrix}\bar{f}_A + K_{A,B}K^{-1}_{B,B}(\bar{f}_{B\mid z_B}-\bar{f}_B)\\ \bar{f}_{B\mid z_B} \end{pmatrix}
$$
并且：
$$
P_{\mid z_B}=K+\left(\begin{array}{c}
K_{A, B} K_{B, B}^{-1} \\
I
\end{array}\right)\left[P_{B \mid z_B}-K_{B, B}\right]\left(\begin{array}{c}
K_{B, B}^{-1} K_{B, A} \\
I
\end{array}\right)^T
$$

### Tracking Non-Stationary Functions

令在时间$t_{k-1}$和$t_k$时刻的空间核分别为$K_{k-1},K_k$。对时间$t_{k-1}$和$t_k$时间段的函数进行建模的空间核为$K$：
$$
K = \begin{pmatrix}K_k&C_{k-1}\\C_{k-1}^T&K_{K-1}\end{pmatrix}
$$
选择$C_{k-1}$使得$K$是半正定的。

我们可以应用AKS算法来计算在时间$t_k$时的后验均值$\bar{f}_k$和协方差$P_k$：
$$
\begin{aligned}
\bar{f}_{k \mid k-1} &=\bar{f}_{k \mid 0}+C_{k-1} K_{k-1}^{-1}\left(\bar{f}_{k-1 \mid k-1}-\bar{f}_{k-1 \mid 0}\right) \\
P_{k \mid k-1} &=K_k+C_{k-1} K_{k-1}^{-1}\left[P_{k-1 \mid k-1}-K_{k-1}\right] K_{k-1}^{-1} C_{k-1}^T
\end{aligned}
$$


通过定义KF过程模型：
$$
G_{k-1} \triangleq C_{k-1}K_{k-1}^{-1}
$$
和KF过程噪声协方差：
$$
Q_k \triangleq K_k - C_{k-1}K_{k-1}^{-1}C_{k-1}^T
$$
我们重现了KF公式。我们重现的第一个公式是后验状态的预测：
$$
\bar{f}_{k\mid k-1} = G_{k-1}\bar{f}_{k-1\mid k-1}
$$
之后，我们重现了先验状态的预测：
$$
\bar{f}_{k\mid 0} = G_{k-1}\bar{f}_{k-1\mid 0}
$$
最后我们得到了后验协方差：
$$
P_{k\mid k-1} = G_{k-1}P_{k-1\mid k-1}G_{k-1}^T + Q_k
$$
一般来说，过程模型$G_{k-1}$为时间差$t_k-t_{k-1}$的函数，因此$G_{k-1} = G(t_k - t_{k-1})$。在$t_k$处的后验均值不应该依赖于它是直接通过$\bar{f}_{k\mid k-1} = G_{k-1}\bar{f}_{k-1\mid k-1}$计算的，还是在中间某一时刻$t_{k^\prime}$计算的，$t_{k-1}<t_{k^\prime}<t_k$。数学上，我们需要：
$$
G(t_k-t_{k-1}) = G(t_k-t_{k^\prime})G(t_{k^\prime}-t_{k-1})
$$

### Applications

#### Tracking Periodic Dynamics



