---
title: NUMPY (一)
summary: Learning Notes (numpy) 
date: '2025-03-25'
authors:
 - william
tags:
 - basic knowledge and skills
---

---
# numpy

import numpy as np

## 直接创建数组

    arr1 = np.array([1, 2, 3, 4])
    print(arr1)  # [1 2 3 4]

## 创建全零数组（默认 float64）

    arr2 = np.zeros((2, 3))  
    print(arr2)

    [[0. 0. 0.]
     [0. 0. 0.]]

## 创建全一数组（int 类型）
    arr3 = np.ones((3, 2), dtype=int)
    print(arr3)
 
    [[1 1]
     [1 1]
     [1 1]]

## 创建未初始化数组（值是随机的）

    arr4 = np.empty((2, 2))  

## np.arange(start, stop, step) - 创建等差数列

    arr5 = np.arange(1, 10, 2)
    print(arr5)   [1 3 5 7 9]

## np.linspace(start, stop, num) - 创建均匀分布的数列

    arr6 = np.linspace(0, 1, 5)  
    print(arr6)   [0.   0.25 0.5  0.75 1.  ]

## 数组属性

其中数据类型int32 和 int64（大）主要区别在于 存储大小 和 表示范围。

    arr = np.array([[1, 2, 3], [4, 5, 6]])

    print(arr.shape)    (2, 3) - 形状：2行3列
    print(arr.dtype)    int64 - 数据类型（可能因环境不同而变化）
    print(arr.size)     6 - 总元素个数
    print(arr.ndim)     2 - 维度数

    arr_int = np.array([1, 2, 3], dtype=np.int32)
    arr_float = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    arr_bool = np.array([True, False, True], dtype=np.bool_)

## 数据类型转换

```
arr = np.array([1.2, 2.5, 3.8])
arr_int = arr.astype(np.int32)  # 转换为整数
print(arr_int)   [1 2 3]
```
```
arr_str = arr.astype(str)  # 转换为字符串
print(arr_str)   ['1.2' '2.5' '3.8']
```

## 索引&切片

```
arr = np.array([10, 20, 30, 40, 50])

print(arr[0])    # 10 - 获取第一个元素
print(arr[1:3])  # [20 30] - 获取索引 1 到 2 的元素
print(arr[:3])   # [10 20 30] - 获取前 3 个元素
print(arr[-1])   # 50 - 获取最后一个元素
```
```
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr[0, 1])  # 2 - 取第 0 行第 1 列
print(arr[:, 1])  # [2 5] - 取所有行的第 1 列
print(arr[1, :])  # [4 5 6] - 取第 1 行的所有元素
```
```
arr = np.arange(1, 7)  # [1 2 3 4 5 6]

arr_reshaped = arr.reshape((2, 3))  
print(arr_reshaped)
# [[1 2 3]
#  [4 5 6]]
```
## broadcasting

```
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([1, 2, 3])

result = arr1 + arr2  # 广播 arr2，使其形状与 arr1 匹配
print(result)
# [[2 4 6]
#  [5 7 9]]
```
## 基础运算

```
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print(np.add(arr1, arr2))      # [5 7 9]  - 加法
print(np.subtract(arr1, arr2)) # [-3 -3 -3]  - 减法
print(np.multiply(arr1, arr2)) # [4 10 18] - 逐元素乘法
print(np.divide(arr1, arr2))   # [0.25 0.4 0.5] - 逐元素除法
```
```
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])

逐元素相乘
print(mat1 * mat2)
# [[ 5 12]
#  [21 32]]

矩阵乘法（dot / matmul）
print(np.dot(mat1, mat2))  # 或者 np.matmul(mat1, mat2)
# [[19 22]
#  [43 50]]
```
```
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(np.sum(arr))        # 21 - 所有元素求和
print(np.mean(arr))       # 3.5 - 均值
print(np.median(arr))     # 3.5 - 中位数
print(np.std(arr))        # 1.7078 - 标准差
```

	•	axis=0：按 列 进行运算（跨行计算）。
	•	axis=1：按 行 进行运算（跨列计算）。

```
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sum(arr, axis=0))  # [5 7 9] - 按列求和
print(np.sum(arr, axis=1))  # [6 15]  - 按行求和
```
## np.linalg 矩阵运算

求逆

```
A = np.array([[4, 7], [2, 6]])
A_inv = np.linalg.inv(A)
print(A_inv)
# [[ 0.6 -0.7]
#  [-0.2  0.4]]
```

计算行列式

```
det_A = np.linalg.det(A)
print(det_A)  # 10.0
```

计算特征值和特征向量

```
eig_vals, eig_vecs = np.linalg.eig(A)
print(eig_vals)  # [8. 2.] - 特征值
print(eig_vecs)  
# [[ 0.8944 -0.7071]
#  [ 0.4472  0.7071]] - 特征向量
```

求解线性方程组 Ax = b

```
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

x = np.linalg.solve(A, b)
print(x)  # [2. 3.]
```
## 随机数生成

```
np.random.rand()（生成 0 到 1 的随机数）、
np.random.randn()（生成标准正态分布随机数）、
np.random.randint()（生成指定范围内的随机整数）。
随机种子 np.random.seed(3) 的作用(3作为举例)，用于确保结果可重复
```

## 文件输入输出

保存和加载 NumPy 二进制文件

```
arr = np.array([[1, 2, 3], [4, 5, 6]])
np.save("array.npy", arr)  # 保存
loaded_arr = np.load("array.npy")  # 加载
print(loaded_arr)
```

保存和加载文本文件

```
np.savetxt("array.txt", arr, delimiter=',')  # 逗号分隔
loaded_txt = np.loadtxt("array.txt", delimiter=',')
print(loaded_txt)
```
