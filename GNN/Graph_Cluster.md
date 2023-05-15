# Graph Clustering with Graph Neural Networks

本文中，我们提出的模型为深度模块网络(Deep Modularity Networks)(DNoN)。一个用于 GNN 的无监督聚类模块，允许以端到端可微分的方式优化聚类分配，具有强大的经验性能。

## Preliminaries

图$G=(V,E)$定义在一组节点$V = (v_1,\cdots,v_n), |V|=n$并且边$E\subseteq V\times V, |E|=m$。邻接矩阵$A$，度$d_i := \sum_{j=1}^n A_{ij}$。我们想要找到图分割函数$\mathcal{F}: V\mapsto \{1,\cdots,k\}$将节点$V$分割为$k$类，$V_i = \{v_j : F(v_j) = i\}$。

### Graph Clustering Quality Functions

经典的聚类对象都是离散的因此不利于梯度下降优化，因此DMoN使用谱近似。

### Spectral Modularity Maximization

令$\mathrm{C}\in 0,1^{n\times k}$为类别分配矩阵，$d$为度向量。则，模块矩阵(modularity matirx)$\mathrm{B}$定义为$\mathrm{B=A-\frac{dd^T}{2m}}$，模块$\mathcal{Q}$可以被写为：

$$
\mathcal{Q} = \frac{1}{2m}\text{Tr}(\mathrm{C^TBC})
$$

最大化$\mathcal{Q}$的$\mathrm{C}$为模块矩阵$\mathrm{B}$的前$k$个特征向量。尽管$\mathrm{B}$是稠密的，迭代特征值求解器可以利用$\mathrm{B}$是稀疏矩阵$\mathrm{A}$和秩$1$矩阵$-\frac{dd^T}{2m}$的和，意味着矩阵向量乘法$\mathrm{Bx}$可以高效计算为：

$$
\mathrm{Bx = Ax - \frac{d^Txd}{2m}}
$$

### Graph Neural Networks

令$\mathrm{X}^0 \in \mathbb{R}^{n\times s}$为初始节点特征并且$\mathrm{\tilde{A}=D^{-\frac{1}{2}}AD^{-\frac{1}{2}}}$为归一化连接矩阵，第$t$层$\mathrm{X}^{t+1}$为：

$$
\mathrm{X}^{t+1} = \text{SeLU}(\mathrm{\tilde{A}X^tW + XW_{\text{skip}}})
$$

我们对经典GCN做了两点改变，首先，我们去除了self-loop creation(应该指的是自己跟自己连接)而使用一个$\mathrm{W}_{\text{skip}}\in \mathbb{R}^{s\times s}$可训练的跳跃连接(trainable skip connection)，第二我们将ReLU非线性函数用SeLU替代，为了更好的收敛。

## Method

### DMoN: Deep Modularity Networks

我们GNN modularity包含两部分：

1. 来编码类别分配矩阵$\mathrm{C}$的架构
2. 最优化的目标函数

我们定义$\mathrm{C}$：

$$
\mathrm{C} = \text{softmax}(\text{GCN}(\mathrm{\tilde{A},X}))
$$

我们定义我们优化的函数为：

$$
\mathcal{L}_{\text{DMoN}}(\mathrm{C;A}) = \underbrace{-\frac{1}{2m}\text{Tr}(\mathrm{C^TBC})}_{\text{modularity}} + \underbrace{\frac{\sqrt{k}}{n}\left\|\sum_i \mathrm{C}_i^T\right\|_F-1}_{\text{collapse regularization}}
$$

