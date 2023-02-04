# Stan Reference Manual

## Character Encoding

所有的`stan`文件必须采用`ASCII`编码，注释可以采用`//`、`#`或者`/*   */`。

## Includes

我们可以对`stan`程序进行分文件编写，如我们将下列程序命名为`my-std-normal.stan`：

~~~stan
functions {
    real my_std_normal_lpdf(vector y) {
        return -0.5 * y' * y;
    }
}
~~~

我们可以使用`include`来引入一个`stan`文件，类似于`C++`中的语法：

~~~stan
#include my-std-normal.stan
paramters {
	real y;
}
model {
	y ~ my_std_normal();
}
~~~
我们也可以将`#include`语句下载非初始化行中。例如我们可以在`#include`前面添加空格：
~~~stan
   #include my-std-normal.stan
parameters {
    // ...
}
~~~
> 单数`#`和`include`之间不能有空格，否则会被当做注释。

## Comments
`stan`支持`C++`类型的注释，单行用`//`，多行用`/*   */`。

## Data Types and Declarations

### Overview of data type

#### Primitive types

`stan`含有两种`primitive types`，对于连续值的`real`类型和对于整数的`int`类型。两者都为**标量类型**。

#### Complex types

`stan`提供复数类型`complex`，它的实部和虚部都为`real`类型。

#### Vector and matrix types

`stan`提供三种实值矩阵数据类型：`vector`对于列向量、`row_vector`对于行向量，`matrix`对于矩阵。

`stan`也提供了三种复数值的矩阵数据类型：`complex_vector`对于列向量、`complex_row_vector`对于行向量，`complex_matrix`对于矩阵类型。

#### Array types

数组类型的元素可以上面提到的任何数据类型，如：

~~~stan
array[10] real x;
array[6, 7] matrix[3, 3] m;
array[12, 8, 15] complex z;
~~~

#### Constrained data types

我们可以对上述数据类型添加约束，除了`complex`类型的其他基本数据类型都可以添加`lower`和`upper`约束：

~~~stan
int<lower=1> N;
real<upper=0> log_p;
vector<lower=-1, upper=1>[3] rho;
~~~

对于矩阵和向量还可以添加其他约束：

+ `simplex`：单纯形、`unit_vector`：单位向量、`ordered`：升序、`positive_ordered`：升序且元素都为正数。
+ `corr_matrix`和`cov_matrix`分别表示相关系数矩阵和协方差矩阵

### Primitive numerical data types

#### Integers

`stan`中所有的整数类型都占$4$个字节，取值范围为$-2^{31}\sim2^{31}-1$，所以我们需要控制取值范围。整数类型在做除法是会四舍五入。

#### Reals

`stan`使用$8$个字节来表示实数，取值范围大致是$\pm 2^{1022}$。还有三种特殊的`real`类型：

1. 对于错误的条件产生not-a-number：
2. 溢出的正无限
3. 溢出的负无限

##### Not-a-number

如果一个实值函数的参数为not-a-number，那么它的返回值或者抛出的异常都会是not-a-number，如果对于布尔返回值的比较操作，如果一个参数是not-a-number，返回值都会是$0$(false)。

##### Infinite values

正无限值比除了他自己和not-a-number以外的任何数字都大，负无限同理。有限数和无限数相加得到无限数，无限数除以有限数得到合适符号的无限数，有限数除以无限数得到零。有限数除以零得到正无限数，两个无限数相除和相减都得到not-a-number。

### Complex numerical data type

构建复数类型：

~~~stan
complex z = 2 - 1.3i;
real re = get_real(z);  // re has value 2.0
real im = get_imag(z);  // im has value -1.3
~~~

也可以通过`to_complex`函数：

~~~stan
vector[K] re;
vector[K] im;
// ...
for (k in 1:K) {
  complex z = to_complex(re[k], im[k]);
  // ...
}
~~~

### Scalar data types and variable declarations

我们将`int`、`real`和`complex`类型合称为`scalar`类型。

#### Affinely transformed real

`real`变量可能定义在已经使用仿射变换：$x\mapsto \mu+\sigma*x$变换后的空间上，其中$\mu$称为offset、$\sigma>0$称为multiplier。

例如：

~~~stan
parameters {
  real<offset=mu, multiplier=sigma> x;
}
model {
  x ~ normal(mu, sigma);
}
~~~

#### Expressions as bounds and offset/multiplier

我们可以使用表达式来作为约束，例如：

~~~stan
data {
 real lb;
}
parameters {
   real<lower=lb> phi;
}

data {
   int<lower=1> N;
   array[N] real y;
}
parameters {
   real<lower=min(y), upper=max(y)> phi;
}
~~~

#### Declaring optional variables

变量的长度可以由布尔常数确定，例如：

~~~stan
data {
  int<lower=0, upper=1> include_alpha;
  // ...
}
parameters {
  vector[include_alpha ? N : 0] alpha;
  // ...
}
~~~



在上面的程序中，如股票`include_alpha`为$1$，则`alpha`的大小为$N$，否则为$0$(没有元素)。



## Statements



