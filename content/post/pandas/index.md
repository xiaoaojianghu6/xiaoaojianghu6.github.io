---
title: PANDAS (一)
summary: Learning Notes (pandas) 
date: '2025-03-26'
authors:
 - william
tags:
 - basic knowledge and skills
---

---

# pandas

```
import pandas as pd
```

---
1. 数据导入和导出

从 CSV、Excel、SQL、JSON 导入数据

CSV 文件
```
df = pd.read_csv("patients.csv")  # 读取 CSV 文件
print(df.head())  # 查看前 5 行数据
```
Excel 文件
```
df = pd.read_excel("patients.xlsx", sheet_name="Sheet1")
print(df.info())  # 显示数据结构信息
```
SQL 数据库
```
import sqlite3

conn = sqlite3.connect("hospital.db")  # 连接 SQLite 数据库
df = pd.read_sql("SELECT * FROM patients", conn)  # 读取 SQL 数据
conn.close()
```
JSON 文件
```
df = pd.read_json("patients.json")
print(df.columns)
```
---

**数据导出**

导出为 CSV
```
df.to_csv("patients_cleaned.csv", index=False)  # 不包含索引列
```
导出为 Excel
```
df.to_excel("patients_cleaned.xlsx", index=False, sheet_name="Cleaned Data")
```
导出到 SQL
```
conn = sqlite3.connect("hospital.db")
df.to_sql("patients_cleaned", conn, if_exists="replace", index=False)  # 替换表
conn.close()
```
医疗领域应用

	•	导入患者电子病历（EHR），包括检查结果、诊断信息等。

	•	导出清洗后的数据 供医生或研究人员分析，例如导出实验数据供统计分析。

⸻

2. 数据结构

Series（一维数据）
```
s = pd.Series([23, 45, 18, 67], index=["A", "B", "C", "D"])
print(s)
```
应用：用于存储 单个患者的生物标志物值（如血糖、血压）。

DataFrame（二维数据）
```
data = {
    "PatientID": [101, 102, 103],
    "Age": [34, 29, 42],
    "Diagnosis": ["Diabetes", "Hypertension", "Healthy"]
}
df = pd.DataFrame(data)
print(df)
```
应用：存储 患者的个人信息和诊断。

索引和切片
```
print(df.loc[0])  # 通过标签索引
print(df.iloc[1])  # 通过位置索引
print(df.loc[:, "Diagnosis"])  # 获取特定列
```
多层索引（MultiIndex）
```
arrays = [["A", "A", "B", "B"], ["One", "Two", "One", "Two"]]
index = pd.MultiIndex.from_tuples(list(zip(*arrays)))
df_multi = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], index=index, columns=["X", "Y"])
print(df_multi)
```

应用：在医疗试验数据中，按 实验组 & 受试者编号 进行索引。

---

3. 数据清洗

缺失值处理
```
df = pd.DataFrame({"Age": [25, None, 35], "Blood Pressure": [120, 130, None]})
print(df.isnull())  # 检测缺失值
df_cleaned = df.dropna()  # 删除含有缺失值的行
df_filled = df.fillna(df.mean())  # 用均值填充缺失值
```

应用：填补 缺失的血压或实验数据。

重复值处理
```
df = pd.DataFrame({"PatientID": [1, 2, 2, 3], "Name": ["Alice", "Bob", "Bob", "Charlie"]})
df_unique = df.drop_duplicates()  # 删除重复行
```
应用：去重 患者数据，防止数据重复。

异常值处理
```
df["Age"] = df["Age"].replace(999, df["Age"].median())  # 替换异常年龄值
```
应用：修正 错误输入的年龄（如 999）。

数据类型转换
```
df["Age"] = df["Age"].astype(int)  # 转换为整数类型
```
字符串处理
```
df["Diagnosis"] = df["Diagnosis"].str.lower()  # 统一转换为小写
```
---

4. 数据转换

apply()：对整列数据应用函数
```
df["Age Group"] = df["Age"].apply(lambda x: "Young" if x < 30 else "Old")
```
应用：划分年龄组（如 18-30 岁为 “Young”）。

map()：用于 Series 进行元素映射
```
df["Risk Level"] = df["Diagnosis"].map({"diabetes": "High", "hypertension": "Medium", "healthy": "Low"})
```
应用：将 疾病类别转换为风险等级。

applymap()：用于 DataFrame 逐元素转换
```
df[["Age", "Blood Pressure"]] = df[["Age", "Blood Pressure"]].applymap(lambda x: x if x > 0 else None)
```
应用：将 异常值替换为 NaN。

---

**数据重塑**

df.melt()：宽表转长表
```
df = pd.DataFrame({
    "PatientID": [101, 102],
    "BloodPressure_Systolic": [120, 130],
    "BloodPressure_Diastolic": [80, 85]
})
```
```
df_long = df.melt(id_vars=["PatientID"], var_name="Measurement", value_name="Value")
print(df_long)
```
应用：将 血压数据展开，适用于时间序列分析。

df.pivot()：长表转宽表
```
df_wide = df_long.pivot(index="PatientID", columns="Measurement", values="Value")
print(df_wide)
```
应用：将 实验数据重新组织，便于可视化。

---

**透视表**
```
df = pd.DataFrame({
    "Doctor": ["Dr. A", "Dr. A", "Dr. B", "Dr. B"],
    "Diagnosis": ["Diabetes", "Hypertension", "Diabetes", "Hypertension"],
    "Count": [10, 15, 20, 25]
})
```
```
pivot_table = pd.pivot_table(df, values="Count", index="Doctor", columns="Diagnosis", aggfunc="sum")
print(pivot_table)
```
应用：统计每个医生诊断的患者人数。

---