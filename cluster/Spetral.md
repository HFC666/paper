# A Tutorial on Spectral Clustering

## Similarity graphs

给定数据点$x_1,\cdots,x_n$以及相似度$s_{ij}\ge 0$。相似度图$G=(V,E)$，图中的每一个顶点$v_i$表示一个数据$x_i$，两个节点是连接的如果相似度$s_{ij}$为正的或者大于某个阈值，边的权重为$s_{ij}$。我们的目标是对图进行划分，使组内的相似度最高，而组外的相似度低。

### Graph notation

令$G=(V,E)$为无向图，节点$V = \{v_1,\cdots,v_n\}$。我们假设$G$是有权图，权重为非负值$w_{ij}\ge 0$，将其记录为加权邻接矩阵$W=(w_{ij})_{i,j=1,\cdots,n}$。如果$w_{ij}=0$则说明节点$v_i,v_j$之间没有边。节点$v_i\in V$的度定义为：

$$
d_i = \sum_{j=1}^n w_{ij}
$$

给定节点集的子集$A\subset V$，我们定义其补集$V\backslash A$为$\bar{A}$。我们定义指示向量$\mathbb{1}_A = (f_1,\cdots,f_n)^\prime \in \mathbb{R}^n$，其中$f_i=1$如果$v_i\in A$，否则为$0$，为了方便我们写作$\{i\mid v_i \in A\}$。对于两个不相交的集合$A,B$，我们定义：

$$
W(A,B) := \sum_{i\in A, j\in B}w_{ij}
$$

我们考虑两种不同的测度来描述子集$A\subset V$的"大小"：

$$
\begin{aligned}
|A| &:= A\text{中节点的数量}\\
\text{vol}(A) &:=\sum_{i\in A}d_i
\end{aligned}
$$

子集$A$是连接的如果对于其中的任意两个点都存在路径相连，$A$是连接的成分(connected component)如果$A$是连接的并且与$A$之外的点不是连接的。非空集合$A_1,\cdots,A_k$形成一个分割如果$A_i\cap A_j = \varnothing, A_1\cup \cdots \cup A_j=V$。

### Different similarity graphs

**The $\epsilon-$neighborhood graph**：我们对于距离小于$\epsilon$的点对进行连接，一般为无权图。

**$k-$nearest neigbor graphs**：如果$v_i$是$v_j$的$k$邻域我们就将其连接，但是这样定义得到的是有向图，我们可以将其转换为无向图。一种方法是如果$v_j$位于$v_i$的$k$邻域或$v_i$位于$v_j$的$k$邻域我们就将其连接，这被称为*k-nearest neighbor graph*，另一种是两者都位于各自的邻域才将其连接，称为*mutual k-nearest neighbor graph*，之后根据其距离将边进行加权。

**The fully connected graph**：这种是全连接的，权重通过高斯相似函数计算$\exp(-\|x_i-x_j\|^2/(2\sigma^2))$。

## Graph Laplacians and their basic properties

Laplacians矩阵是谱聚类的主要工具。我们假设图$G$是无向加权图，规定特征值都是递增的，我们称前$k$个特征向量对应$k$个最小的特征值。

### The unnormalized graph Laplacian

非归一化图拉普拉斯矩阵定义为：
$$
L = D-W
$$

**性质一($L$的性质)**：矩阵$L$满足下列性质：

1. 对于任何向量$f\in\mathbb{R}^n$，我们有：$$f^\prime Lf = \frac{1}{2}\sum_{i,j=1}^nw_{ij}(f_i-f_j)^2$$
2. $L$为对称和半正定的。
3. $L$最小的特征值为$0$，与之对用的特征向量为常数一向量$\mathbb{1}$。
4. $L$有$n$个非负、实值特征值$0=\lambda_1\le \lambda_2 \le \cdots \le \lambda_n$

**性质二(Number of connected components and the spectrum of $L$)**：令$G$为一个具有非负权重的无向图。则$L$的$0$特征值对应的特征向量的重数(multiplicity)$k$对应于连接成分$A_1,\cdots,A_k$的数量。特征值$0$的特征空间的基为$\mathbb{1}_{A_1},\cdots,\mathbb{1}_{A_k}$。

### The normalized graph Laplacians

有两个矩阵被称为归一化图拉普拉斯矩阵：
$$
\begin{aligned}
L_{\text{sym}} &:= D^{1/2}LD^{-1/2} = I - D^{-1/2}WD^{-1/2}\\
L_{\text{rw}} &:= D^{-1}L = I - D^{-1}W
\end{aligned}
$$

**性质三(Properties of $L_{\text{sym}}$ and $L_\text{rw}$)**：归一化拉普拉斯满足下列性质：

1. 对于任何$f\in \mathbb{R}^n$我们有：$$f^\prime L_{\text{sym}}f = \frac{1}{2}\sum_{i,j=1}^nw_{ij}\left(\frac{f_i}{\sqrt{d_i}} - \frac{f_j}{\sqrt{d_j}}\right)$$
2. $\lambda$是$L_{\text{rw}}$特征向量$u$对应的特征值当且仅当$\lambda$为$L_{\text{sym}}$的特征向量$w=D^{1/2}u$对应的特征值。
3. $\lambda$是$L_{\text{rw}}$特征向量$u$对应的特征值当且仅当$\lambda,u$为特征问题$Lu=\lambda Du$的解。
4. $0$为$L_{\text{rw}}$的特征值，对应的特征向量为$\mathbb{1}$；$0$为$L_{\text{sym}}$的特征值对应的特征向量为$D^{1/2}1$。
5. 两者都为半正定矩阵，含有$n$个非负特征向量$0=\lambda_1\le \cdots\le \lambda_n$。

**性质四(Number of connected components and spectra of $L_{\text{sym}}$ and $L_{\text{rm}}$)**：令$G$为一个有非负权重的无向图。则特征值$0$对用的重数$k$，对于$L_{\text{rw}},L_{\text{sym}}$来说，等于连接组成$A_1,\cdots,A_k$的数量。对于$L_{\text{rw}}$，$0$的特征空间由$\mathrm{1}_{A_i}$张成，对于$L_{\text{sym}}$，$0$的特征空间由$D^{1/2}1_{A_i}$张成。

## Graph cut point of view

我们可以将此问题看作是一个最小分割(mincut)问题，组内权重大，不同组的边权重小。

符号：$W(A,B) := \sum_{i\in A, j\in B}w_{ij},\bar{A}$为$A$的补集。给定$k$个子集，mincut问题最小化：
$$
\text{cut}(A_1,\cdots,A_k) := \frac{1}{2}\sum_{i=1}^kW(A_i,\bar{A}_i)
$$

两种应用比较多的目标函数为RatioCut和归一化分割Ncut：

$$
\begin{aligned}
\text{RatioCut}(A_1,\cdots, A_k) &:= \frac{1}{2}\sum_{i=1}^k \frac{W(A_i,\bar{A}_i)}{|A_i|} = \sum_{i=1}^k\frac{\text{cut}(A_i,\bar{A}_i)}{|A_i|}\\
\text{NCut}(A_1,\cdots, A_k) &:= \frac{1}{2}\sum_{i=1}^k \frac{W(A_i,\bar{A}_i)}{\text{vol}(A_i)} = \sum_{i=1}^k\frac{\text{cut}(A_i,\bar{A}_i)}{\text{vol}(A_i)}
\end{aligned}
$$

但是求解此问题是NP-hard问题，但是我们可以利用谱聚类进行求解，我们会看到relaxing Ncut会得到归一化谱聚类，relaxing RatioCut会得到非归一化谱聚类。

## Random walks point of view

随机游走为随机过程，我们可以将其看作是在相似图的点上进行随机游走，在同一个类中停留的时间较长，而在组间的游走的概率较低。定义状态转移矩阵$P = (p_{ij},{i,j=1,\cdots,n}),p_{ij}:=w_{ij}/d_i$，即

$$
P = D^{-1}W
$$
