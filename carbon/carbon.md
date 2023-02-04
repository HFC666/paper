## Assessment of Carbon Emission and Carbon Sink Capacity of China’s Marine Fishery under Carbon Neutrality Target

carbon sinks：碳汇

net carbon emissions：净碳排放

decarbonization space：脱碳空间

log-mean decomposition index method

#### 碳排放的计算

$$
\mathrm{C} = \sum_{m=1}^7 \mu_m\cdot P_m\cdot h
$$

其中$\mathrm{C}$是煤炭消耗量，$m$为渔船的不同作业方式，$\mu$是不同作业方式渔船的燃油消耗系数，$P$为不同作业方式渔船的主机功率，$h$为标准煤的燃料油换算系数，为$1.4571$。
$$
\mathcal{Q}_c = \mathcal{Q}_E\cdot F_C\cdot C\cdot \delta
$$
其中$\mathcal{Q}_c$为碳量，$\mathcal{Q}_E$为有效氧化级分(effective oxidation fraction)，取值为$0.982$，$F_C$为每吨标准煤的碳量，取值$0.73257$，$\delta$为在相同的热能下，燃料油和煤燃烧产生的二氧化碳的比率，取值为$0.813$。
$$
\mathcal{Q}_{co_2} = \mathcal{Q}_C\cdot \omega
$$
其中$\mathcal{Q}_{co_2}$为二氧化碳的排放量，$\omega$为碳转换为二氧化碳的常数，取值为$3.67$。



#### 碳汇的计算

海洋渔业的碳汇的主要来源是贝类和藻类养殖

##### 海水养殖贝类的碳汇估计

根据贝类碳收支方程，贝类总膳食碳(total dietary carbon)TDC可以被分解为粪便碳(fecal carbon)FC，排泄碳(excretion carbon)EC，呼吸碳(respiration carbon)RC和生长碳(growth carbon)GC。
$$
\begin{gathered}
C_i^s=P_i^{s h} \cdot R_i^s \cdot w_i^s \cdot\left(1-\varepsilon_i\right) \\
C_i^{s t}=P_i^{s h} \cdot R_i^{s t} \cdot w_i^{s t} \\
C_i^{P O C}=\left(\frac{C_i^s}{1-\varepsilon_i}+C_i^{s t}\right) \cdot \frac{F C+E C}{G C} \cdot \gamma^{P O C} \\
T C^{s h}=\sum_i\left(C_i^s+C_i^{P O C}\right) \\
T C_{C O_2}^{s h}=T C^{s h} \cdot \omega
\end{gathered}
$$
贝类养殖的碳汇可以分为两部分：贝类碳汇$C_i^s$和在贝类生长过程中颗粒有机物释放形成的碳汇$C_i^{POC}$。$\omega$表示将碳转换为二氧化碳的常数。软组织中的碳$C_i^{st}$被认为不是碳汇。$i$表示贝类的种类，$P_i^{st}$表示贝类产量(湿重)，$R_i^s$表示贝类的干重比例，$w_i^s$表示贝壳的碳含量。$1-\epsilon_i$表示壳碳源换算系数，$\epsilon_i$表示表示贝壳中有机碳或海洋沉积物中的碳占贝壳总碳的比例。$R_i^{st}$表示软组织的干重的比例，$w_i^{st}$表示软组织的碳含量。$C_i^{POC}$依赖于贝类生长碳数据，在碳平衡方程中采用$\frac{FC+EC}{GC}=1$的标度关系进行测量。$\gamma^{POC}$为碳汇和POC的转化率。

##### 海洋藻类的碳汇估计

海水养殖藻类的总碳汇可以用以下公式测量：
$$
\begin{aligned}
C_j^a &= P_j^{al}\cdot w_j^a\\
C_j^{POC} &= C_j^a\cdot \frac{\alpha}{1-\alpha-\beta}\cdot \gamma^{POC}\\
C_j^{DOC} &= C_j^a\cdot \frac{\beta}{1-\alpha-\beta}\cdot \gamma^{DOC}\\
TC^{al} &= \sum_j\left(C_j^a + C_j^{POC}+C_j^{DOC}\right)\\
TC_{CO_2}^{al}  &= TC^{al}\cdot \omega
\end{aligned}
$$
海水养殖藻类总碳汇$TC^{al}$包含藻类身体碳汇$C_j^a$，藻类通过释放POC和DOC形成的碳汇$C_j^{POC}$和$C_j^{DOC}$，$\omega$为将碳转换为二氧化碳的比例，$j$表示藻类的种类。$P_j^{al}$为藻类产量(干重)，$w_j^a$为藻类碳含量。$\alpha$和$\beta$分别表示藻类生长过程中释放的POC和DOC在光合作用生产力中的份额。$\gamma^{POC}$和$\gamma^{DOC}$分别反映了生物释放的POC和DOC最终转化为碳汇的速率。

##### 净碳排放的计算

$$
C_{net} = \mathcal{Q}_{CO_2} - \left(TC_{CO_2}^{sh} + TC_{CO_2}^{al}\right)
$$

#### 净碳排放影响因素的分解分析



