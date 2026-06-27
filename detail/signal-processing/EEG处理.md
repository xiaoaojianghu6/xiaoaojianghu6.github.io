# eeg-frequency-domain-analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目报告

本项目是一个基于 Python 的脑电图 (EEG) 信号处理实验。主要实现了两个核心功能：
1.  **使用快速傅里叶变换 (FFT) 对处理后的信号进行频域分析**，并将其分解为不同的脑波频带 (Delta, Theta, Alpha, Beta, Gamma)。
2.  **使用独立成分分析 (ICA) 去除 EEG 信号中的常见伪影** (如眼电、肌电干扰)。


## 主要功能

* **数据加载与预处理**: 从 JSON 文件加载多通道 EEG 数据，并创建 MNE-Python 数据结构。
* **频域分析**:
    * 实现带通滤波器以分离特定的脑波节律。
    * 应用快速傅里叶变换 (FFT) 来分析信号的功率谱密度 (PSD)。
* **伪影去除**: 应用独立成分分析 (ICA) 算法自动检测并移除眼动、眨眼等噪声成分。
* **可视化**:
    * 将原始信号分解为 Delta, Theta, Alpha, Beta, Gamma 波并进行可视化。
    * 对比 ICA 处理前后的 EEG 信号，直观展示降噪效果。
    * 以图表形式展示不同脑波频带的功率对比，量化 ICA 的有效性。
    

## 编译环境

* **编程语言**: Python 3.x
* **核心库**:
    * [MNE-Python](https://mne.tools/stable/index.html): 用于 EEG 数据处理和 ICA 分析的专业库。
    * [NumPy](https://numpy.org/): 科学计算的基础。
    * [SciPy](https://scipy.org/): 用于信号处理，特别是滤波器和 FFT。
    * [Matplotlib](https://matplotlib.org/): 用于数据可视化。

## 结果



### 1. 脑波频带分解

经过 ICA 清理后的信号可以被分解为不同的神经节律。下图展示了原始信号及其分解出的五个主要脑波频带。

![脑波频带分解图](results/figures/1.png)

其中，Beta波段信号受到较为严重的肌电干扰:

![能量图](results/figures/2.png)

### 2. ICA 伪影去除效果

考虑 ICA 算法，也许可以有效地识别并去除信号中的噪声成分。

提取出9个components：
![9compnents](results/figures/3.png)

剔除0，1，8，三个疑似伪影后对比通道在独立成分分析处理前后的信号波形，可以看到对干扰有一定滤除效果。


![ICA前后的信号对比](results/figures/4.png)

下图进一步量化了 ICA 的效果，对比了处理前后各脑波频带的功率。肌电伪影（主要干扰beta波）（带通后，更高频率大部分滤掉了）在去噪后其功率显著下降。

![ICA前后各频带功率对比](results/figures/5.png)



## 许可 (License)
本项目采用 [MIT 许可证](LICENSE)。