## NN-aided UKF

$$
\begin{aligned}
x_{k+1} &= f(x_k) + n_k\\
y_k &= h(x_k) + v_k
\end{aligned}
$$

我们用$e_k$表示真实模型和**先验已知**的数学模型$\hat{f}(x_k)$，即$e_k = f(x_k) - \hat{f}(x_k)$。当$e_k-g(x_k,w_k)\rightarrow 0$，误差$e_k$被很好地近似，我们有：
$$
\begin{aligned}
x_{k+1} &= \hat{f}(x_k) + g(x_k, w_k)\\
w_{k+1} &= w_k
\end{aligned}
$$
我们将状态(state)表示为$x_k^a = [x_k^T\quad w_k^T]^T$，则上式变为：
$$
x_{k+1}^a = \begin{bmatrix}x_{k+1}\\ w_{k+1}  \end{bmatrix} = \begin{bmatrix}
    \hat{f}(x_k) + g(x_k, w_k)\\w_k
\end{bmatrix}
\triangleq f^a(x_k,w_k)
$$

