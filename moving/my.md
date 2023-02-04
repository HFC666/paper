### My Model

状态变量：

+ $x$：经度
+ $y$：纬度
+ $v_x$：经向速度
+ $v_y$：纬向速度
+ $a_t$：切向加速度
+ $a_n$：向心加速度

那么我们的状态转移矩阵变为：
$$
A = \begin{bmatrix}1&0&T&0&\frac{1}{w_t}T\cos\theta_t+\frac{1}{w_t^2}\sin\theta_t - \frac{1}{w_t^2}\sin(\theta_t+w_tT)&\frac{1}{w_t^2}\cos\theta_t-\frac{1}{w_t}T\sin\theta_t-\frac{1}{w_t^2}\cos(\theta_t+w_tT)\\0&1&0&T&\frac{1}{w_t^2}\cos\theta_t - \frac{1}{w_t}T\sin\theta_t-\frac{1}{w_t^2}\cos(\theta_t+w_tT)&\frac{1}{w_t^2}\sin(\theta_t+w_tT)-\frac{1}{w_t^2}\sin\theta_t-\frac{1}{w_t}T\cos\theta_t\\0&0&1&0&\frac{1}{w_t}\cos\theta_t-\frac{1}{w_t}\cos(\theta_t+w_tT)&\frac{1}{w_t}\sin(\theta_t+w_tT)-\frac{1}{w_t}\sin\theta_t\\0&0&0&1&\frac{1}{w_t}\sin(\theta_t+w_tT)-\frac{1}{w_t}\sin\theta_t&\frac{1}{w_t}\cos(\theta_t+w_tT)-\frac{1}{w_t}\cos\theta_t\\0&0&0&0&1&0\\0&0&0&0&0&1\end{bmatrix}
$$

$$
s_t = \begin{bmatrix}x_t\\y_t\\v_{x_t}\\v_{y_t}\\a_{t_t}\\a_{n_t}\end{bmatrix}
$$

所以
$$
s_{t+1} = As_t + Q_t
$$
其中
$$
w_k = \frac{a_{n_k}}{\sqrt{{v_x}_t^2}+{v_y}_t^2}
$$


观测状态矩阵
$$
H = \begin{bmatrix}1&0&0&0&0&0\\0&1&0&0&0&0\\0&0&1&0&0&0\\0&0&0&1&0&0\end{bmatrix}
$$
观测数据
$$
o_t = \begin{bmatrix}x_t\\y_t\\v_{x_t}\\v_{y_t}\end{bmatrix} 
$$
所以
$$
o_t = Hs_t + R_t
$$


下面就可以用卡尔曼滤波了。

