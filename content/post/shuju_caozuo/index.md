---
title: 数据操作学习笔记（基于pytorch框架）
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

# 预处理

创建一个人工数据集，存储在CSV（逗号分隔值）文件 ../data/house_tiny.csv中

```
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

从创建的CSV文件中加载原始数据集，导入pandas包并调用read_csv函数

```
import pandas as pd

data = pd.read_csv(data_file)
print(data)

   NumRooms Alley   Price
0       NaN  Pave  127500
1       2.0   NaN  106000
2       4.0   NaN  178100
3       NaN   NaN  140000
```

处理缺失值：用均值替代
```
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

   NumRooms Alley
0       3.0  Pave
1       2.0   NaN
2       4.0   NaN
3       3.0   NaN
```

处理缺失值：单独分为一类
```
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

   NumRooms  Alley_Pave  Alley_nan
0       3.0           1          0
1       2.0           0          1
2       4.0           0          1
3       3.0           0          1
```