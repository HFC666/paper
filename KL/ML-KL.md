## Machine Learning meets Kalman Filtering

我们考虑时空过程(spatio-temporal processes)。$a$表示时间和空间。因此，为了不失一般性，我们可以写作$f(a)=f(x,t)$。相应地，定义域$\mathscr{A}$可以被分解为$\mathscr{A}:=\mathscr{X}\times \mathbb{R}_+$，$\mathscr{X}$和$\mathbb{R}_+$分别表示空间和时间域。

### Spectral factoriation of random processes

考虑协方差为$h(\tau)$的平稳随机过程$f(t)$。多亏了Wiener-Khinchin定理，随机过程的功率谱密度(PSD)等价于他的协方差的傅里叶变换：
$$
S(\omega) := \mathscr{F}[h(\tau)](\omega)
$$
更进一步，在特殊情况下当$S=S_r$为$2r$阶有理的(不知道什么意思)，多亏了谱分解，它的PSD可以写作$S_r(\omega) = W(i\omega)W(-i\omega)$，其中：
$$
W(\mathbf{i} \omega)=\frac{b_{r-1}(\mathbf{i} \omega)^{r-1}+b_{r-2}(\mathbf{i} \omega)^{r-2}+\cdots+b_0}{(\mathbf{i} \omega)^r+a_{r-1}(\mathbf{i} \omega)^{r-1}+\cdots+a_0}
$$
上述式子等价于下面的连续时间状态空间模型：
$$
\begin{cases}
\dot{s}_t &= Fs_t + Gw_t\\
z_t &= Hs_t
\end{cases}
$$
其中$w_t \sim \mathscr{N}(0, I)$，模型矩阵等价于
$$
\begin{aligned}
F &=\left[\begin{array}{ccccc}
0 & 1 & 0 & \ldots & 0 \\
0 & 0 & 1 & \ldots & 0 \\
& & & \ddots & \\
0 & 0 & 0 & \ldots & 1 \\
-a_0 & -a_1 & -a_2 & \ldots & -a_{r-1}
\end{array}\right], \quad G=\left[\begin{array}{c}
0 \\
0 \\
\vdots \\
0 \\
1
\end{array}\right], \\
H &=\left[\begin{array}{lllll}
b_0 & b_1 & b_2 & \ldots & b_{r-1}
\end{array}\right],
\end{aligned}
$$
初始状态为$s_0\sim \mathscr{N}(0,\Sigma_0)$，其中$\Sigma_0$为李亚普诺夫方程$FX+XF^T+GG^T=0$的解。

