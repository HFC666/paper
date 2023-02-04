## Regression Models
### Linear regression
简单的线性回归模型只包含一个因变量：
$$
    y_n \sim \text{normal}(\alpha+\beta X_n,\sigma)
$$
其模型用`stan`描述为：
~~~stan
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}

parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
}

model {
    y ~ normal(alpha + beta * x, sigma);
}
~~~
我们考虑有多个因变量的情况：
~~~stan
data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  matrix[N, K] x;   // predictor matrix
  vector[N] y;      // outcome vector
}
parameters {
  real alpha;           // intercept
  vector[K] beta;       // coefficients for predictors
  real<lower=0> sigma;  // error scale
}
model {
  y ~ normal(x * beta + alpha, sigma);  // likelihood
}
~~~

### The QR reparameterization
在上面的例子中，我们可以将模型写为$\eta = x\beta$，其中$\eta \in \mathrm{R}^N, x\in \mathbb{R}^{N\times K},\beta \in \mathbb{R}^K$。假设$N\ge K$，矩阵$x$可以被QR分解为正交矩阵$Q$和上三角矩阵$R$，使得$x = QR$。
函数`qr_thin_Q`和`qr_thin_R`实现`thin`QR分解：
对于矩阵$\mathrm{A}\in \mathbb{R}^{m\times n},m>n$，其`thin`QR分解为：
$$
\begin{aligned}
\mathbf{A} &=\mathbf{Q}\left(\begin{array}{c}
\mathbf{R}_{1} \\
0
\end{array}\right) \\
&=\left(\mathbf{Q}_{1} \mathbf{Q}_{2}\right)\left(\begin{array}{c}
\mathbf{R}_{1} \\
0
\end{array}\right) \\
&=\mathbf{Q}_{1} \mathbf{R}_{1}
\end{aligned}
$$
其中$R_1$为一个$n\times n$的上三角矩阵，$Q_1$为有正交列的$m\times n$矩阵。
那么我们可以对$x$进行分解：$x = Q^\star R^\star$，其中$Q^\star = \sqrt{n-1}Q,R^\star = R/\sqrt{n-1}$，则$\eta = x \beta = Q^\star R^\star \beta$。我们令$\theta = R^\star \beta$，则$\eta = Q^\star \theta, \beta = (R^\star)^{-1}\theta$。我们的`stan`程序为：
~~~stan
data {
    int<lower=0> N;
    int<lower=0> K;
    matrix[N, K] x;
    vector[N] y;
}

transformed data {
    matrix[N, K] Q_ast;
    matrix[K, K] R_ast;
    matrix[K, K] R_ast_inverse;

    // thin and scale the QR decomposition
    Q_ast = qr_thin_Q(x) * sqrt(N - 1);
    R_ast = qr_thin_R(x) / sqrt(N - 1)l
    R_ast_inverse = inverse(R_ast);
}

parameters {
    real alpha;
    vector[K] theta;
    real<lower=0> sigma;
}

model {
    y ~ normal(Q_ast * theta + alpha, sigma);
}

generated quantities {
    vector[K] beta;
    beta = R_ast_inverse * theta;
}
~~~
利用QR分解方法可以使采样的速度加快。
### Logistic and probit regression
逻辑回归模型可以描述为：
~~~stan
data {
    int<lower=0> N;
    vector[N] x;
    array[N] int<lower=0, upper=1> y;
}

parameters {
    real alpha;
    real beta;
}

model {
    y ~ bernolli_logit(alpha + beta * x);
}
~~~
逻辑回归是一种输出为二元变量的广义线性回归模型，logit连接函数定义为：
$$
\text{logit}(v) = \log\left(\frac{v}{1-v}\right)
$$
