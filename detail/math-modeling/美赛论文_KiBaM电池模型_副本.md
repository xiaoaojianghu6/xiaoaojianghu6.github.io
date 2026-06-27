---
type: project
status: completed
aliases: [美赛论文, KiBaM电池模型, MCM2026ProblemA]
tags: [项目, 数据, 技术]
date_created: 2026-02-01
date_modified: 2026-06-23
related_concepts: []
related_people: [William-Liu]
---

# 美赛论文: KiBaM 双井电池模型预测手机续航

MCM/ICM 2026 Problem A(Team 2623972), **Honorable Mention 奖**: Beyond Coulomb Counting -- A Dual-Well Kinetic Battery Model for Accurate Smartphone Time-to-Empty Prediction。原文 PDF 已转 md(本文件)。

## 摘要

传统电池管理依赖简单库仑计数, 无法捕捉倍率容量效应/恢复现象/环境影响, 导致"电量焦虑"。本文构建基于物理的电池建模框架。

## 方法(四问)

1. **连续时间模型**: KiBaM 双井框架(available/bound wells 动力学转移)+ 二阶等效电路 + 集总热耦合(Arrhenius)+ SEI 老化(√τ 扩散限制)+ 组件功耗模型(OLED/CPU DVFS/5G NR/Wi-Fi/后台/湿度漏电)。
2. **TTE 预测**: 四种真实场景 + 蒙特卡洛 500 次。OCV 预测 RMSE=89.93mV(R²=0.8412), TTE 预测 RMSE=42.3min(R²>0.85), TTE 分布对数正态, 均值 8.2h, 95%CI[6.1,10.9]h。
3. **敏感性分析**: 显示亮度(|Sβ|≈0.65)与 CPU 利用率(|Sᵤ|≈0.52)影响最大; 多参数交互超线性协同。移除 KiBaM 误差增 35-45%, 验证倍率容量/恢复效应必要性。
4. **建议**: 自适应亮度(降 20% 延长 TTE 13%)、后台限制、Wi-Fi 优于 5G(0.8-1.0W vs 2.5-4.5W)、温度感知。框架可推广笔记本/平板/电动车。

## 关键词

KiBaM, Battery aging, Time-to-empty prediction, Equivalent circuit model。

## 我的贡献

物理建模核心(KiBaM 双井 + 等效电路 + 热 + 老化多模型耦合)+ 蒙特卡洛不确定性量化 + 敏感性/协同分析。

## 关联

- 本人: [[William-Liu]]
