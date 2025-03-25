---
title: 数据操作学习笔记
summary: Learning Notes (Data Operation)
date: '2025-03-24'
authors:
 - william
tags:
 - basic knowledge and skills
---
# 数据操作

    import torch

---
## 初始化一个张量

创建一个行向量（0～11）

    x = torch.arange(12)

改变一个张量的形状而不改变元素数量和元素值

    X = x.reshape(3, 4)

全0；全1；

    torch.zeros((2, 3, 4))
    torch.ones((2, 3, 4))

创建一个形状为（3,4）的张量。 
其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样。

    torch.randn(3, 4)

**赋值定义**

    torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
注意外面有两层括号，一层小括号➕一层中括号

---

## 运算操作

torch.exp(x) 函数是用于计算输入张量x中每个元素的自然指数（即e的指数）的函数
其中，注意指数表达：

    2.7183e+00    2.7183 * 10^0 = 2.7183
    2.9810e+03    2.9810 * 10^3 = 2981.0

    X = torch.arange(12, dtype=torch.float32).reshape((3,4))
    Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
---
连结concatenate

    torch.cat((X, Y), dim=0), 沿纵轴连接（3）
    torch.cat((X, Y), dim=1)  沿横轴连接（4）

逻辑运算（True=1，false=0）

    X==Y（X>Y、、、）
    tensor([[0, 1, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 0]])

求和

    X.sum()
得到单元素张量

---
对于形状不同的张量的运算遵循broadcasting mechanism

    a=tensor([[0],
              [1],
              [2]]),
            
    b=tensor([[0, 1]])
    a+b
    tensor([[0, 1],
        [1, 2],
        [2, 3]])
---
索引

    X[-1], 第一维（行）最后一个元素（如二维矩阵取最后一行）
    X[1:3]，取得是1，2（第二第三行）

写入

    X[1   , 2  ] = 9  第二行   ，第三列     =9
    X[0:2 ,  : ] = 9  一二两行 ，（的每一列均）=9

---
## 节省内存

    后续计算中没有重复使用X， 使用X[:] = X + Y或X += Y来减少内存开销。
---
