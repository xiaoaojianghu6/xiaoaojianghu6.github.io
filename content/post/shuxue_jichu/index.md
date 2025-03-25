---
title: 基础数学使用笔记（一）
summary: Learning Notes (math) 
date: '2025-03-25'
authors:
 - william
tags:
 - basic knowledge and skills
---

---
# 线性代数

注：矩阵中，第一维是行，第二维是列； 例如3×7→3行；  axis=0(行)；axis=1(列)
注意区别torch.Size([4])则表达一维向量，如[0,1,2,3]

转置

    A.T

Hadamard积（形状相同的矩阵）

    A * B

 ！[png](output1.png)

降维
求x所有元素的和

    x.sum（）

沿着轴0（保持轴0不变）求和

    x.sum（axis=0）

保持所有维度不变，如:(沿axis=0)则把第一行加到第二行，把第一第二行加到第三行...

    X.cumsum(axis=0)

平均值

    X.mean()   （ =  X.sum() / X.numel() ）

点积（相同位置的按元素乘积的和）

    torch.dot(x, y)

矩阵向量积（例如先将矩阵A用它的行向量表示，然后与x点积）

    torch.mv(A, x)

矩阵矩阵（考虑A的行向量和B的列向量点积）

！[png](output2.png)

    torch.mm(A, B)

范数(norm) →向量范数是将向量映射到标量的函数f，对向量x满足：绝对值缩放，三角不等式，f(x)非负。
向量的范数是表示一个向量有多大（size）不涉及维度，专指分量的大小

！[png](output3.png)

有点像距离的概念➡️欧几里得距离是一个L2范数
一般地，Frobenius范数（Frobenius norm）是矩阵所有元素平方和的平方根：
计算L2范数或Frobenius范数：

    u = torch.tensor([3.0, -4.0])
    torch.norm(u)

    tensor(5.)

L1范数表示为向量元素的绝对值之和：

    u = torch.tensor([3.0, -4.0])
    torch.abs(u).sum()

    tensor(7.)







