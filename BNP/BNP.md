---
title: 狄利克雷过程
categories: 论文阅读
tags: [机器学习, 无参贝叶斯]
update: 2022-7-9
state: 完成
---
## 狄利克雷过程

### Introduction

考虑下列问题，假设我们用高斯混合模型来做聚类。假设我们的数据为$x_1,x_2,\cdots,x_N$，那我们的似然函数的对数为：
$$
\sum_{i=1}^N\log\sum_{l=1}^K\alpha_i\mathcal{N}(\mu_i,\sigma_i^2)
$$
但是我们需要事先确定聚类的个数，但是在很多情况下聚类的个数并不是那么容易确定，我们需要从数据中学习到聚类的个数。一个方法是我们将聚类的数目$K$也作为一个参数，那么我们的参数$\theta = (K,\theta_1,\sigma_1,\cdots,\theta_K,\sigma_K)$。我们的优化问题变为：
$$
\hat{\theta} = \arg\max_{\theta}\sum_{i=1}^N\log\sum_{l=1}^K\alpha_i\mathcal{N}(\mu_i,\sigma_i^2)
$$
但是我们很容易发现当$K=N$，$\mu_i$为数据的值，$\sigma_i=0$的时候似然函数达到最大，即每一个数据都是一个类，这不是我们想要的。
我们的一个方法是假设每个数据$x_i$都来自于参数为$\theta_i$的一个分布，而$\theta_i\sim H(\theta)$，其中分布$H(\theta)$为连续分布，但是这样会参在新的问题：因为$H(\theta)$为连续分布，所以$P(\theta_i=\theta_j)=0,i\neq j$。所以这样每个数据来自的分布都不一样，又回到了之前提到的$K=N$的问题了。所以我们令$\theta$来自一个离散分布$G(\theta)$，而$G\sim \text{DP}(\alpha,H)$，其中DP表示狄利克雷过程，而$H$为之前的连续分布，$\alpha>0$为常数，其反映$G$的离散程度，越大表示离散程度越小，当$\alpha\rightarrow0$时，$G$离散程度最大，为一个点；当$\alpha\rightarrow \infty$时，$G\approx H$。

> $H$也不一定是连续的，其被称为base measure。

注意这里的$G$为一个random measure，我们每次从$\text{DP}(\alpha,H)$中采样得到的不是一个数值，而是一个分布。假设我们采集到的分布为：

![](https://raw.githubusercontent.com/HFC666/image/master/img/DP1.png)

> 其中$G_1,G_2$为两次采样产生的分布，上面的棍子表示概率密度，其和为$1$。我们将其分为$a_1,a_2,\cdots,a_d$等$d$个区域，其中每个区域的总概率密度符合狄利克雷分布，即
> $$
> (G(a_1),G(a_2),\cdots,G(a_d))\sim \text{Dir}(\alpha H(a_1),\alpha H(a_2),\cdots,\alpha H(a_d))
> $$
> 这就是狄利克雷过程的定义。

关于狄利克雷分布：
$$
P(x_1,\cdots,x_i,\cdots,x_K)\sim \text{Dir}(\alpha_1,\cdots,\alpha_i,\cdots,\alpha_K)
$$
则
$$
\begin{aligned}
\mathbb{E}[x_i]  &= \frac{\alpha_i}{\sum_k \alpha_k}\\
\text{Var}[x_i]&=\frac{\alpha_i(\sum_k\alpha_k-\alpha_i)}{(\sum_k\alpha_k)^2(\sum_{k}\alpha_k+1)}
\end{aligned}
$$
将其带入到狄利克雷过程中，得：
$$
\mathbb{E}[G[a_i]] = \frac{\alpha H(a_i)}{\sum_k\alpha H(a_k)} = H(a_i)
$$

$$
\text{var}[G[a_i]] = \frac{\alpha H(a_i)(\alpha-\alpha H(a_i))}{\alpha^2(\alpha+1)} = \frac{H(a_i)(1-H(a_i))}{\alpha+1}
$$

我们之前说过关于$\alpha$的性质。现在我们来看一下，均值(期望)与$\alpha$无关，当$\alpha\rightarrow \infty$是，方差趋近于$0$，这说明不管我们怎么划分，在每个$a_i$处，$G(a_i)=H(a_i)$，说明$G(x)=H(x)$，即$G(x)$是连续的是最不离散的版本；如果$\alpha=0$，则$\text{Var}=H(a_i)(1-H(a_i))$，这正是伯努利分布的方差，因此在每个划分上我们都可以用一根棍来表示它们(包括不划分)，这是最离散的版本。

### Construction

那我们如何构建$G$呢？即如何从$H$中采样得到$G$呢？我们采用`stick-breaking construction`的方法来产生$G$。其构建方式为：

1. 从$H$中采样得到$\theta_1$，即$\theta_1\sim H$
2. 采样$\beta_1\sim\text{Beta}(1,\alpha)$
3. 权重$\pi_1=\beta_1$
4. 采样$\theta_2\sim H$
5. 采样$\beta_2\sim \text{Beta}(1,\alpha)$
6. 权重$\pi_2 = (1-\pi)\beta_2$
   以此类推，这样权重$\pi_2$相当于是从取完$\pi_1$剩下的权重中取得。

我们在看一下关于$\alpha$得一些内容，当$\alpha=0$时，$\mathbb{E}(\beta_i)=1,\text{Var}(\beta_i)=0$，所以一次就把权重全部采完，即只产生一个样本，对应于最离散的情况，而当$\alpha=\infty$时，$\mathbb{E}(\beta_i)=0$，相当于连续分布的情况。我们常将此采样方法写为$\pi \sim \text{GEM}(\alpha)$。

### Property

下面我们再回顾一下。对于狄利克雷过程，有如下性质：
$$
G\sim \text{DP}(\alpha,H)\Leftrightarrow (G(a_1),\cdots,G(a_n))\sim \text{DIR}(\alpha H(a_1),\cdots,\alpha H(a_n)),\quad \text{for any partitions }a_1,\cdots,a_n
$$
总结之前讲过的，我们有：
$$
G\sim \text{DP}(\alpha,H)
$$

$$
\theta_1,\cdots,\theta_N\sim G
$$

$$
X_i\sim F(\theta_i)
$$

其图模型为：

![](https://raw.githubusercontent.com/HFC666/image/master/img/DP2.png)

下面我们研究一下$G$的后验分布$P(G\mid \theta_1,\cdots,\theta_N)$：
$$
P(G\mid \theta_1,\cdots,\theta_N)\propto P(\theta_1,\cdots,\theta_N\mid G)P(G)
$$
在研究之前我们先看一个关于狄利克雷分布和多项式分布的例子：
假设
$$
\begin{aligned}
p_1,\cdots,p_N&\sim\text{DIR}(\alpha_1,\cdots,\alpha_N)\\
n_1,\cdots,n_N&\sim\text{Mult}(p_1,\cdots,p_N)
\end{aligned}
$$
那么
$$
\begin{aligned}
P(p_1,\cdots,p_N\mid n_1,\cdots,n_N)&\propto\left(\frac{\Gamma(\sum_{i=1}^N\alpha_i)}{\prod_{i=1}^N\Gamma(\alpha_i)}\prod P_i^{\alpha_i-1}\right)\left(\frac{(\sum_{i=1}^Nn_i)!}{n_1!\cdots n_N!}\prod_{i=1}^NP_i^{n_i}\right)\\&\propto \prod_{i=1}^NP_i^{\alpha_i+n_i-1}\\ &= \text{DIR}(\alpha_1+n_1,\cdots,\alpha_N+n_N)
\end{aligned}
$$

### Posterior

有了前面的指示，下面我们来看一下后验分布：
对于任何划分
$$
\begin{aligned}
P(G(a_1),\cdots,G(a_K)\mid \theta_1,\cdots,\theta_K)&\propto P(\theta_1,\cdots,\theta_K\mid G(a_1),\cdots,G(a_K))P(G)\\
&=\text{Mult}(n_1,\cdots,n_K)\text{DIR}(\alpha H(a_1),\cdots,\alpha H(a_K))\\
&= \text{Dir}(n_1+\alpha H(a_1),\cdots,n_K+\alpha H(a_K))\\
&= \text{DP}\left(\alpha+n,\frac{\alpha H+\sum_{i=1}^K\delta_{\theta_i}}{\alpha+n}\right)
\end{aligned}
$$
其中$n=\sum_{i=1}^K n_i，G=\sum_{i=1}^\infty\pi_i\delta_{\theta_i}$。最后一步是怎么来的呢？
所以我们得到的后验分布为：
$$
P(G(a_1),\cdots,G(a_K))\sim \text{Dir}(n_1+\alpha H(a_1),\cdots,n_K+\alpha H(a_K))
$$
根据之前讲过的狄利克雷过程的性质，狄利克雷过程的第一个参数是对应的狄利克雷分布的测度和，而第二个参数为归一化后的一个概率分布，理解为狄利克雷分布的参数除以归一化系数。所以第一个参数为：$\sum_{i=1}^K n_i+\alpha H(a_i) = \alpha+n$，而第二个参数为$\frac{\alpha H+\sum_{i=1}^K\delta_{\theta_i}}{\alpha+n}$，其中$\alpha+n$为归一化参数，而分子$\sum_{i=1}^N\delta_{\theta_i}(a_j)$实际上就是表示$n_j$。
我们再看一下得到的分布：
$$
\frac{\alpha H+\sum_{i=1}^N\delta_{\theta_i}}{\alpha+n} = \frac{\alpha}{\alpha+n}H+\frac{\sum_{i=1}^N\delta_{\theta_i}}{\alpha+n}
$$
为一个连续的分布加上一个离散的分布，这被称为`spike and slab`。

### Predictive distribution

什么是预测分布呢？预测分布为：
$$
\begin{aligned}
P(X_i\mid X_{-i}) &= \int_w P(X_i,w\mid X_{-i})dw\\
&= \int_w P(X_i\mid w,X_{-i})P(w\mid X_{-i})dw\\
&= \int_wP(X_i\mid w)P(w\mid X_{-i})dw
\end{aligned}
$$
其中$X_{-i}$表示去除第$i$项后的$X$。
对于狄利克雷过程，我们想要求得的为：
$$
P(\theta_i\mid \theta_{-i}) = \int_G P(\theta_i\mid G)P(G\mid \theta_{-i})dG
$$
由此可以看出，我们的预测分布可以看作是后验分布$\theta_i$在后验分布$P(G\mid \theta_{-i})$下的期望，根据狄利克雷分布的性质，其期望为
$$
\frac{\alpha}{\alpha+n}H+\frac{\sum_{i=1}^N\delta_{\theta_i}}{\alpha+n}
$$
所以其预测分布等于后验分布。

$\theta_1,\theta_2,\cdots$预测分布的序列被称为`Blackwell-MacQueen urn scheme`。这个名字来源于一个隐喻。特别地，在$\Theta$中的每一个值都是唯一的颜色，并且抽样$\theta\sim G$来给球上色。另外我们有一个盒子来装之前看过的球。起初在盒子里没有球，我们从$H$中取颜色，$\theta_1\sim H$，给球上色并将其放在盒子里。在之后的步骤中，如在$n+1$步中，我们要么以$\frac{\alpha}{\alpha+n}$抽取一个新颜色($\theta_{n+1}\sim H$)，给球染色并将其放到盒子中，或者以概率$\frac{n}{\alpha+n}$从盒子中取出一个球，将新球涂成它的颜色(从经验分布中抽样)并放到盒子里。

`Blackwell-MacQueen urn scheme`可以被用来证明DP的存在。我们可以在序列$\theta_1,\theta_2,\cdots$上构建分布，通过迭代地在给定$\theta_1,\cdots,\theta_{i-1}$的条件下采样$\theta_i$。对于$n\ge1$令
$$
P(\theta_1,\cdots,\theta_n) = \prod_{i=1}^nP(\theta_i\mid \theta_1,\cdots,\theta_{i-1})
$$
可以得到这个随机序列是无限可交换的。也就是说，对于每一个$n$，生成$\theta_1,\cdots,\theta_n$的概率等于以任何顺序采样得到它们的概率。

下面我们来证明一下，令$I_k$表示第$k$类的索引，$N_k$表示第$k$类的样本数，那么在第$k$类的样本的上述关于$\theta$的概率为：
$$
\frac{\alpha\cdot1\cdot2\cdots(N_k-1)}{(I_{k,1}-1+\alpha)(I_{k,2}-1+\alpha)\cdots(I_{k,N_k)}-1+\alpha)}
$$
第一项是因为我们第一次到新的类$k$，所以概率为$\frac{\alpha}{I_{k,1}-1+\alpha}$，第二项是因为$k$已经出现了，所以概率为$\frac{1}{(I_{k,2}-1+\alpha)}$，以此类推。对于所有的类：
$$
p(\theta_{1:N}) = \prod_{k=1}^K\frac{\alpha(N_k-1)!}{(I_{k,1}-1+\alpha)(I_{k,2}-1+\alpha)\cdots(I_{k,N_k}-1+\alpha)}
$$
注意所有$I_k$的并为所有的索引，我们将索引合并，得：
$$
p(\theta_{1:N}) = \frac{\alpha^K\prod_{k=1}^K(N_k-1)!}{\prod_{i=1}^N(i-1+\alpha)}
$$
所以很明显看出来是无限可交换的。



现在`de Finetti's theorem`说明对于任何无限可交换序列$\theta_1,\theta_2,\cdots$存在一个随机分布$G$使得序列可以被分解为独立同分布地从下列采样：
$$
P(\theta_1,\cdots,\theta_n) = \int\prod_{i=1}^n G(\theta_i)dP(G)
$$
在我们的设置中，随机分布$P(G)$的先验正是狄利克雷过程$\text{DP}(\alpha,H)$，因此DP存在。

### Clustering, Partitions and the Chinese Restaurant Process

DP的离散性质也暗示了聚类的特性。现在我们假设$H$是光滑的，因此所有的重复值都由于DP的离散性质而不是$H$自身。因此采样得到的值有重复的，令$\theta_1^\star,\cdots,\theta_m^\star$为$\theta_1,\cdots,\theta_n$去除重复值后的结果并且$n_k$为$\theta_k^\star$重复的次数。预测分布可以被等价地写为：
$$
\theta_{n+1}\mid \theta_1,\cdots,\theta_n\sim \frac{1}{\alpha+n}\left(\alpha H+\sum_{k=1}^m n_k\delta_{\theta_k^\star}\right)
$$


我们可以通过查看由聚类引起的划分来进一步研究DP的聚类属性。$\theta_1,\cdots,\theta_n$去除重复值后将对集合$[n]=\{1,\cdots,n\}$分区引进了聚类使得在每一个类$k$中，$\theta_i$取相同的值$\theta_k^\star$。

分区的分布被称为中国餐馆过程(CRP)。在这个情境下我们有一个中国餐馆，里面有无穷多个桌子，每个桌子可以坐无穷多个人。第一个人进入餐馆坐在第一个位置，第二个人可以坐在第一个人的位置或者坐在新的位置。一般地，第$n+1$个人要么以正比于$n_k$的概率坐在已经有人的位置$k$，或者以正比于$\alpha$的概率坐在新的位置。

我们也可以估计聚类数目的期望。假设共有$n$个观测，对于$i\ge1$，观测$\theta_i$以$\frac{\alpha}{\alpha+i-1}$的概率取新的值，所以聚类数$m$的期望为：
$$
\mathbb{E}[m\mid n] = \sum_{i=1}^n\frac{\alpha}{\alpha+i-1}\in \mathcal{O}(\alpha \log n)
$$


因为$\theta$为离散值，具有相同$\theta$值得数据表示属于同一类，我们可以用$Z$来表示属于哪一类。即计算$P(Z_i\mid Z_{-i})$。有多少类只与参数$\alpha$有关，而$\theta$的位置(值)则与$H$有关。



我们下面计算：
$$
P(Z_i=m\mid Z_{-i}) = \frac{P(Z_i=m,Z_{-i})}{P(Z_{-i})}
$$
我们如何将其与狄利克雷过程结合起来呢？将其与狄利克雷过程结合起来很难，因为狄利克雷过程有无限多的项，我们可以假设其外$K$项，然后再将$K$取无穷。我们用如下方法：
$$
\begin{aligned}
P(Z_i=m\mid Z_{-i}) &= \frac{P(Z_i=m,Z_{-i})}{P(Z_{-i})}\\
&=\frac{\int_{P_1,\cdots,P_K}P(Z_i=m,Z_{-i}\mid P_1,\cdots,P_K)\text{DIR}(\alpha/K,\cdots,\alpha/K)dP}{\int_{P_1,\cdots,P_K}P(Z_{-i}\mid P_1,\cdots,P_K)\text{DIR}(\alpha/K,\cdots,\alpha/K)dP}
\end{aligned}
$$
关于积分的计算就用到我们之前的多项式分布和狄利克雷分布共轭的知识了：
$$
\begin{aligned}
&\int_{p_1,\cdots,p_K}P(n_1,\cdots,n_K\mid p_1,\cdots,p_K)P(p_1,\cdots,p_K\mid \alpha_1,\cdots,\alpha_K)dP\\
&=\frac{n!}{n_1!\cdots n_K!}\frac{\Gamma(\sum\alpha_i)}{\prod\Gamma(\alpha_i)}\int_{p_1,\cdots,p_K}\prod_{i=1}^K p_i^{n_i+\alpha_i-1}dp\\
&=\frac{n!}{n_1!\cdots n_K!}\frac{\Gamma(\sum\alpha_i)}{\prod\Gamma(\alpha_i)}\frac{\prod\Gamma(\alpha_i+n_i)}{\Gamma(\sum\alpha_i+n)}
\end{aligned}
$$
我们将其应用到分子和分母上，首先定义符号：$n_{l,-i}$表示去除第$i$个数据后属于第$l$类的个数。我们知道第二项：
$$
\frac{\Gamma(\sum\alpha_i)}{\prod\Gamma(\alpha_i)}
$$
只与先验有关，所以分子分母都一样。

我们先看第一项，因为在我们的情景下，即便类数相等，每个类的个数相等，但是相同个体不属于同一个类，这两种划分方法得到的第一项的值是相同的，但是在我们的情境下是不同的，所以第一项不应该存在。

第三项代入为：
$$
\frac{\Gamma(\alpha/K+n_{m,-i}+1)\prod_{l\neq i} \Gamma(\alpha/K+n_{l,-i})}{\Gamma(\alpha+n)}\cdot \frac{\Gamma(\alpha+n-1)}{\prod\Gamma(\sum_l \alpha/K+n_{l,-i})}
$$
伽马函数具有如下性质：
$$
\Gamma(x) = (x-1)\Gamma(x-1)
$$
上式化简得到：
$$
\frac{n_{m,-i}+\frac{\alpha}{K}}{n+\alpha-1}
$$


当$K\rightarrow \infty$为
$$
\frac{n_{m,-i}}{n+\alpha-1}
$$
对$m$进行求和得到：
$$
\sum_m\frac{n_{m,-i}}{n+\alpha-1} = \frac{n-1}{n+\alpha-1}\neq1
$$
这与概率密度的定义不同，所以有$\frac{\alpha}{n+\alpha-1}$的概率属于新的一个类，这就是**中国餐馆过程**。

### Dirichlet Process Mixture Models

狄利克雷混合模型可以写为：
$$
\begin{aligned}
&\pi \mid \alpha \sim \operatorname{GEM}(\alpha)\\
&\theta_{k}^{*} \mid H \sim H\\
&z_{i} \mid \pi \sim \operatorname{Mult}(\pi)\\
&x_{i} \mid z_{i},\left\{\theta_{k}^{*}\right\} \sim F\left(\theta_{z_{i}}^{*}\right)
\end{aligned}
$$
狄利克雷混合模型为无限混合模型，指的是具有无限可数个类的混合模型。与事先确定了类的有限混合模型不同，狄利克雷混合模型会根据数据确定聚类的数目。

> 狄利克雷过程为无参贝叶斯方法的一种，为什么是无参？我们理解是狄利克雷过程是分布的先验，而狄利克雷过程采样得到的分布为我们产生数据的分布，在有参数的模型中此分布有参数(感觉像废话)，而狄利克雷过程产生的分布无法用参数表示故为无参数模型。我们将分布作为概率的一部分，并且存在自己的先验分布，也可以让我们通过数据来自动调节分布的复杂度。