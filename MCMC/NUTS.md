---
title: NUTS采样
categories: 论文阅读
tags: [机器学习, 蒙特卡罗]
update: 2022-8-15
state: 未完成
---

## The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo

虽然HMC避免了采样的随机性，但是其对人为确定的参数是敏感的：步长大小$\epsilon$和步数$L$。例如，如果步数$L$过小，那么就与随机采样无异，而步数过大收敛的速度就会变慢。所欲我们开发出NUTS作为HMC的拓展，不需要我们事先确定$L$和$\epsilon$。

