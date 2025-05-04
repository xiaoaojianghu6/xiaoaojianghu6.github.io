---
authors:
- william
date: '2025-05-01'
summary: ' '
tags: [ML]
title: 全流程梳理（初）
---

# 全流程库

[模版库](https://www.notion.so/1e6db3c8daf880329d59f4c9204e5560?pvs=21)

# 导入库

---

```python
import pandas as pd  #加载，操作
import numpy as np  #计算

from sklearn.model_selection import train_test_split  #数据集划分

from sklearn.preprocessing import StandardScaler, OneHotEncoder  #标准化；独热编码
from sklearn.impute import SimpleImputer   # 缺失值处理
from sklearn.decomposition import PCA       # 降维
from sklearn.metrics import accuracy_score  # 性能评估

from sklearn.linear_model import LogisticRegression  #逻辑回归
from sklearn.linear_model import LinearRegression  #线性回归
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  #决策树，回归树
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb  # XGBoost库，用于调用XGBoost模型
import lightgbm as lgb  # LightGBM库，用于调用LightGBM模型

from sklearn.datasets import ...#自带数据集，如load_breast_cancer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #评估指标

import torch  # PyTorch核心库
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
from torch.utils.data import DataLoader, TensorDataset  # 数据加载和处理

import torchvision                  # 计算机视觉工具包
from torchvision import transforms  # 图像预处理模块

import matplotlib.pyplot as plt
import seaborn as sns
```

```python
# 示例：调用XGBoost
xgb_model = xgb.XGBClassifier()  # 用于分类任务的XGBoost模型
# 或者
xgb_reg_model = xgb.XGBRegressor()  # 用于回归任务的XGBoost模型

# 示例：调用LightGBM
lgb_model = lgb.LGBMClassifier()  # 用于分类任务的LightGBM模型
# 或者
lgb_reg_model = lgb.LGBMRegressor()  # 用于回归任务的LightGBM模型

# 示例：调用Scikit-learn的梯度提升树
gbc_model = GradientBoostingClassifier()  # 用于分类任务的梯度提升树模型
# 或者
gbr_model = GradientBoostingRegressor()  # 用于回归任务的梯度提升树模型
```

关于SK-learn自带数据集：

[自带数据集](https://www.notion.so/1e6db3c8daf880819b20f0b47280d935?pvs=21)

---

# 导入加载数据

### 从.csv文件或.excel导入

```python
url = 'whatever.csv'  #这是路径
df = pd.read_csv(url)  # .read_excel(url)
X = df.drop('target', axis=1)  # 特征：删除整列（axis=1)target,剩下特征
y = df['target']               # 标签
```

### 从sklearn自带数据集导入示例

```python
cancer = load_breast_cancer() #数据集名称
X = cancer.data  #特征
y = cancer.target  #标签
df = pd.DataFrame(X, columns=cancer.feature_names)
df['dx'] = y

'''
cancer = load_breast_cancer()
这里使用 sklearn.datasets 库的 load_breast_cancer() 方法加载乳腺癌数据集。这个数据集是 sklearn 提供的一个经典分类数据集，包含有关乳腺癌肿瘤的特征信息。

X = cancer.data
X 变量存储数据集中所有样本的特征信息，不包括目标变量（标签）。cancer.data 是一个 numpy 数组，每一行代表一个样本，每一列代表一个特征。

y = cancer.target
y 变量存储目标变量，即每个样本的分类标签。数据集中有两个类别：
0 代表恶性（malignant）
1 代表良性（benign）

df = pd.DataFrame(X, columns=cancer.feature_names)
这行代码创建一个 pandas 数据框 df，它的列名 (columns) 采用 cancer.feature_names，即数据集中的所有特征名称。

df['dx'] = y
这里将目标变量 y 添加到数据框 df 的最后一列，并命名为 dx，这样数据框不仅包含样本特征，还包含类别信息。
'''
```

### 从seaborn在线数据集导入示例

```python
df = sns.load_dataset('titanic')
X = df.drop('survived', axis=1)  # 特征集
y = df['survived']  # 目标变量
```

### 利用.wfdb读取生理信号数据集示例

```python
---读所有内容
record = wfdb.rdrecord('/Users/mac/Downloads/data/ECG/mit-bih-arrhythmia-database-1.0.0/' + number, channel_names=['MLII'])
data = record.p_signal.flatten()
rdata = denoise(data=data)

- `wfdb.rdrecord()` 读取编号 `number` 的 ECG 数据记录，并指定导联 `MLII`（II 导联）。
- `record.p_signal.flatten()` 获取原始信号并转换为一维数组。
- `denoise(data)` 调用一个降噪函数（未提供具体定义），可能用于去除噪声，提高信号质量。

标准模式加演示：
record = wfdb.rdrecord('D:/ECG-Data/MIT-BIH-360/100', # 文件所在路径
                       sampfrom=0, # 读取100这个记录的起点，从第0个点开始读
                       sampto=1000, # 读取记录的终点，到1000个点结束
                       physical=False, # 若为True则读取原始信号p_signal，如果为False则读取数字信号d_signal，默认为False
                       channels=[0, 1]) # 读取那个通道，也可以用channel_names指定某个通道;如channel_names=['MLII']

# 转为数字信号
signal = record.d_signal[0:1000]

# 绘制波形
plt.plot(signal)
plt.title("ECG Signal")
plt.show()

---批量读取
type=[]
rootdir = 'mit-bih-arrhythmia-database-1.0.0'           # 设置根路径 
files = os.listdir(rootdir) #列出文件夹下所有的目录与文件
name_list=[]            # name_list=[100,101,...234]
MLII=[]                 # 用MLII型导联采集的人
type={}                 # 标记及其数量
for file in files:
    if file[0:3] in name_list:   # 根据数据库实际情况调整熟知，这里判断的是每个文件的前三个字符
        continue
    else:
        name_list.append(file[0:3])
for name in name_list:      # 遍历每一个数据文件
    if name[0] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:       # 跳过无用的文件
        continue
    record = wfdb.rdrecord(rootdir+'/'+name)  # 读取一条记录（100），不用加扩展名

---读注释
annotation = wfdb.rdann('/Users/mac/Downloads/data/ECG/mit-bih-arrhythmia-database-1.0.0/' + number, 'atr')
Rlocation = annotation.sample
Rclass = annotation.symbol
 `wfdb.rdann()` 读取该记录的注释 (`.atr` 文件)，其中包括 R 波位置 (`annotation.sample`) 和对应的心律失常类型 (`annotation.symbol`)。

---读信号
signal, fields = wfdb.rdsamp('path_to_dataset/sample')

---绘制信号
wfdb.plot_wfdb(record='path_to_dataset/sample', title='ECG Signal')

---长度不一数据剪裁成统一长度
f=360       # 根据不同的数据库进行更改
segmented_len=10        # 目标：裁剪成10s的数据片段
label_count=0
count=0

segmented_data = []             # 最后数据集中的X
segmented_label = []            # 最后数据集中的Y
print('begin!')
for person in MLII:
    k = 0
    while (k+1)*f*segmented_len<=len(whole_signal[0]):    # 只要不到最后一组数据点
        count+=1
        record = wfdb.rdrecord(rootdir + '/' + person, sampfrom=k * f * segmented_len,sampto=(k + 1) * f * segmented_len)  # 读取一条记录（100），不用加扩展名
        annotation = wfdb.rdann(rootdir + '/' + person, 'atr', sampfrom=k * f * segmented_len,sampto=(k + 1) * f * segmented_len)  # 同时读取一条记录的atr文件，扩展名atr
```

[完整实例](https://www.notion.so/1e9db3c8daf880b39ad0cc5fa1c64b0a?pvs=21)

## 导入图片

### Pytorch方法：

```python
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir #路径
        self.mask_dir = mask_dir #路径
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))  # 确保图片和掩码顺序一致
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 可能是灰度图

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整大小
    transforms.ToTensor(),  # 转换为张量
])

# 加载数据集
dataset = ImageMaskDataset("image", "mask", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 取一个样本验证
image_sample, mask_sample = next(iter(dataloader))
print(image_sample.shape, mask_sample.shape)

```

### opencv方法：

```python
import os
import cv2
import numpy as np

image_dir = "image"
mask_dir = "mask"

image_filenames = sorted(os.listdir(image_dir))
mask_filenames = sorted(os.listdir(mask_dir))

images = []
masks = []

for img_file, mask_file in zip(image_filenames, mask_filenames):
    img_path = os.path.join(image_dir, img_file)
    mask_path = os.path.join(mask_dir, mask_file)

    image = cv2.imread(img_path)  # 读取图片
    image = cv2.resize(image, (256, 256))  # 调整大小
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取掩码（灰度图）
    mask = cv2.resize(mask, (256, 256))

    images.append(image)
    masks.append(mask)

# 转换为NumPy数组
images = np.array(images)
masks = np.array(masks)

print(images.shape, masks.shape)

```

---

# 预处理

### 缺失值

```python
（下面的df = pd.read_csv(url)）
# 删除指定特征ABC的字段
df.drop(['A', 'B', 'C'], axis=1, inplace=True)

df.dropna(inplace=True)  # 删除包含缺失值的行
```

```python
# 缺失值处理示例：使用sklearn SimpleImputer替换缺失值（数值型）
# 功能：将DataFrame中缺失值填补为平均值
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)  # X为DataFrame或ndarray

# 填补缺失值
df['A'].fillna(df['A'].median(), inplace=True) #中位数填充
df['E'].fillna(df['E'].mode()[0], inplace=True) #众数填充

```

### 选择需要的特征

```python
from sklearn.feature_selection import SelectKBest, f_classif

# 特征选择示例：选择与目标相关性最强的前K个特征
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X_train, y_train)
selected_features = selector.get_support(indices=True)  # 被选中特征的列索引
```

```python
df = df[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']]
```

### 降维

```python
# 降维示例：主成分分析（PCA）将特征降到指定维度
# 高维数据或冗余特征较多的情况下使用，有助于加速训练和防止过拟合。
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_train_scaled)  # 对标准化后的数据降维
explained_ratio = pca.explained_variance_ratio_  # 每个主成分解释方差比例
```

### 归一化

1. 标准化，转换为均值为 0、标准差为 1 的正态分布

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

1. 最小最大归一化，将数据缩放到指定范围（默认 [0, 1]）

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))  # 可自定义范围
X_normalized = scaler.fit_transform(X)
```

### 编码分类特征

映射分类：

```python
df['sex'] = df['sex'].map({'male': 0, 'female': 1})  # 将性别映射为数值
df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})  # 将登船港口映射为数值

```

独热编码：

```python
# 类别编码示例：独热编码（One-Hot Encoding）
# 功能：将分类特征转化为独热向量
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_cat = np.array([['red'], ['green'], ['blue']])
X_encoded = encoder.fit_transform(X_cat)
'''
[[1. 0. 0.]  # 'red'
 [0. 1. 0.]  # 'green'
 [0. 0. 1.]  # 'blue'
'''
# 如果使用Pandas DataFrame:
X_encoded_df = pd.get_dummies(df['category_column'], prefix='cat')
'''
   cat_blue  cat_green  cat_red
0        0         0       1
1        0         1       0
2        1         0       0
3        0         0       1
'''
```

### 去噪

[小波去噪](https://www.notion.so/1e9db3c8daf8808b9cacd729ffb7581b?pvs=21)

### 数据增强

```python
# 图像增强示例：使用torchvision.transforms执行常见变换
# 功能：随机裁剪、翻转、旋转等增强来扩充数据
data_augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])
```

---

# 划分数据集

```python
from sklearn.model_selection import train_test_split

# 简单划分示例：按8:2比例划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
```

```python
from sklearn.model_selection import KFold, StratifiedKFold

# K折交叉验证示例：将数据分为5折
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in kf.split(X):
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]
```

```python
# 分层抽样示例：用于类别不平衡情况的交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
```

```python
# PyTorch中随机划分数据集示例
from torch.utils.data import random_split

dataset = CustomImageDataset(image_paths, labels, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

---

# 模型定义

### 简单回归模型

```python
# 线性模型示例：使用sklearn逻辑回归作为分类器
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
```

### 决策树

```python
# 树模型示例：使用随机森林分类器
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
```

### CNN

```python
# 简单CNN示例：PyTorch定义一个卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 输入通道数为3（RGB图像），输出通道数为16，卷积核大小3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 第二层卷积
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 全连接层
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # 假设输入图片224x224，池化两次后为56x56
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 卷积+激活+池化
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)            # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model_cnn = SimpleCNN(num_classes=10)
```

### RNN

```python
# 简单RNN示例：PyTorch定义一个双向LSTM
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)  # *2因为双向

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim*2]
        out = out[:, -1, :]    # 取最后时间步作为句子表示
        out = self.fc(out)     # [batch_size, num_classes]
        return out

model_rnn = SimpleRNN(vocab_size=10000, embed_dim=128, hidden_dim=64, num_layers=2, num_classes=5)
```

### transformer

```python
# Transformer示例：使用PyTorch内置Transformer模块
# 功能：构建一个Transformer编码器层的实例
d_model = 512
nhead = 8
num_layers = 6
transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
```

---

# 损失函数&优化器&评估指标（模型定义）

适用于需要梯度下降的神经网络（keras框架）：

```python
model自定义.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy']
                  # metrics: 列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=[‘accuracy’]。
                  )
```

---

# 训练循环设置

基于梯度下降的监督学习任务（keras/sk-learn框架）：

```python
model自定义.fit(X_train, Y_train, epochs=30, batch_size=128, validation_split=RATIO)
```

其他示例：

```python
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()                  # 交叉熵损失函数（常用于分类）
optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)  # Adam优化器
# 训练循环示例：基础版的PyTorch训练流程
num_epochs = 10
for epoch in range(num_epochs):
    model_cnn.train()   # 切换到训练模式
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()           # 清零梯度缓存
        outputs = model_cnn(inputs)     # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()                 # 反向传播计算梯度
        optimizer.step()                # 更新参数

        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
```

---

# 预测评估可视化

### 预测报告

```python
y_pred = 自己定义的模型名.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# 查看训练集准确率（检查过拟合）
y_train_pred = clf.predict(X_train)
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
```

### Train loss

```python
# 绘制训练损失/准确率曲线示例
epochs = range(1, len(train_losses)+1)
plt.plot(epochs, train_losses, 'b-', label='train_loss')
plt.plot(epochs, val_losses, 'r-', label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('train_loss')
plt.show()
```

### 混淆矩阵

```python
# 可视化混淆矩阵
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - 模型名")
plt.show()
```

### ROC

```python
y_score = 模型名.predict_proba(X_test)[:, 1]  # 获取阳性类的预测概率
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'模型名 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - RF')
plt.legend()
plt.grid(True)
plt.show()
```

### 特征重要性

```python
## 特征重要性图
importances = 模型名.feature_importances_
features = df.columns[:-1]
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance (模型名)", fontsize=14)
plt.bar(range(X.shape[1]), importances[indices], color="skyblue", align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
```