### Piecewise cubic Hermite interpolation

假设有$n$各轨迹点，在时间区间$(t_i,t_{i+1})$上按下式进行插值：
$$
\text{lon}(t_i) = a_it_i^3 + b_it_i^2 + c_it_i + d_i
$$
对时间进行求导：
$$
v_{\text{lon}_i} = \frac{d\text{lon}(t_i)}{dt_i} = 3a_it_i^2 + 2b_it_i + c_i
$$
在时间区间$(t_i,t_{i+1})$中，$t_1,t_2,\text{lon}_1,\text{lon}_2,v_{\text{lon}_1},v_{\text{lon}_2}$是已知的，所以我们有：
$$
\begin{aligned}
\text{lon}(t_1) &= a_1t_1^3 + b_1t_1^2 + c_1t_1 + d_1 = \text{lon}_1\\
\text{lon}(t_2) &= a_1t_2^3 + b_1t_2^2 + c_1t_2 + d_1 = \text{lon}_2\\
v_{\text{lon}_1} &= 3a_1t_1^2 + 2b_1t_1 + c_1\\
v_{\text{lon}_2} &= 3a_1t_2^2 + 2b_1t_2 + c_1
\end{aligned}
$$

### Cubic spline interpolation

$$
s_3(t_i) = v_{\text{lon}_i},s_3^\prime(t_0) = v_{\text{lon}_1}^\prime = a_{\text{lon}_1}
$$

