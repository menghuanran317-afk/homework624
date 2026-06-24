# 基于深度学习的家庭电力消耗预测

## 项目简介

本项目基于家庭电力消耗数据（Household Power Consumption）构建时间序列预测模型，研究不同深度学习模型在短期和长期电力负荷预测任务中的表现。

项目主要实现了以下三种模型：

* LSTM（Long Short-Term Memory）
* Transformer
* TCN（Temporal Convolutional Network）

同时结合：

* 历史用电数据
* 天气数据
* 日期特征（周末、节假日、季节等）

提升预测效果。

---

## 项目目标

针对家庭电力消耗数据完成以下两类预测任务：

### 短期预测（Short-Term Forecasting）

输入：

* 过去 90 天数据

输出：

* 未来 90 天用电量

```text
过去90天
    ↓
预测
未来90天
```

---

### 长期预测（Long-Term Forecasting）

输入：

* 过去 90 天数据

输出：

* 未来 365 天用电量

```text
过去90天
    ↓
预测
未来365天
```

---

## 数据集来源

### 1. 家庭电力消耗数据

来源：

UCI Machine Learning Repository

数据集：

Individual Household Electric Power Consumption

链接：

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

数据时间范围：

```text
2006-12 ~ 2010-11
```

主要字段：

| 字段                    | 含义       |
| --------------------- | -------- |
| global_active_power   | 有功功率     |
| global_reactive_power | 无功功率     |
| voltage               | 电压       |
| global_intensity      | 电流强度     |
| sub_metering_1        | 厨房耗电     |
| sub_metering_2        | 洗衣房耗电    |
| sub_metering_3        | 热水器和空调耗电 |

---

### 2. 天气数据

来源：

法国政府开放数据平台

链接：

https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles

使用特征：

| 特征      | 含义        |
| ------- | --------- |
| RR      | 月降雨量      |
| NBJRR1  | 降雨≥1mm天数  |
| NBJRR5  | 降雨≥5mm天数  |
| NBJRR10 | 降雨≥10mm天数 |
| NBJBROU | 大雾天数      |

天气数据与电力数据按照时间进行匹配融合。

---

## 特征工程

### 电力特征

```text
global_active_power
global_reactive_power
sub_metering_1
sub_metering_2
sub_metering_3
voltage
global_intensity
sub_metering_remainder
```

---

### 天气特征

```text
RR
NBJRR1
NBJRR5
NBJRR10
NBJBROU
```

---

### 日期特征

```text
is_weekend
is_holiday
month
day_of_week
day_of_month
day_of_year
week_of_year
season
```

其中：

```text
0 = Winter
1 = Spring
2 = Summer
3 = Autumn
```

---

### 周期编码特征

为了增强模型对周期规律的学习能力，引入：

```text
sin_month
cos_month

sin_dayofweek
cos_dayofweek

sin_dayofyear
cos_dayofyear
```

---

## 数据处理流程

```text
原始电力数据
      ↓
按天聚合
      ↓
天气数据处理
      ↓
日期特征生成
      ↓
数据融合
      ↓
标准化(StandardScaler)
      ↓
滑动窗口构造
      ↓
模型训练
```

---

## 项目结构

```text
ML_Power_Forecast/
│
├── data/
│
│   ├── raw/
│
│   └── processed/
│       ├── daily_power_weather.csv
│       ├── train.csv
│       ├── test.csv
│       ├── train_with_calendar.csv
│       └── test_with_calendar.csv
│
├── src/
│
│   ├── dataset.py
│   ├── train_utils.py
│
│   ├── lstm.py
│   ├── transformer.py
│   ├── tcn.py
│
│   ├── train.py
│   ├── train_transformer.py
│   └── train_tcn.py
│
├── checkpoints/
│
├── figures/
│
└── README.md
```

---

## 模型设计

### 1. LSTM

作为基线模型。

结构：

```text
输入序列
    ↓
LSTM
    ↓
全连接层
    ↓
预测结果
```

特点：

* 能够学习时间依赖关系
* 适用于序列建模
* 长期预测能力有限

---

### 2. Transformer

采用自注意力机制建模长距离依赖关系。

结构：

```text
输入
    ↓
Linear Embedding
    ↓
Position Encoding
    ↓
Transformer Encoder
    ↓
Linear
    ↓
预测结果
```

特点：

* 并行计算能力强
* 长期依赖建模能力强
* 参数量较大

---

### 3. TCN（改进模型）

TCN（Temporal Convolutional Network）采用：

* 因果卷积（Causal Convolution）
* 膨胀卷积（Dilated Convolution）
* 残差连接（Residual Connection）

结构：

```text
输入
    ↓
TCN Block
    ↓
Residual Block
    ↓
全连接层
    ↓
预测结果
```

特点：

* 训练稳定
* 参数较少
* 对小规模数据集表现较好

---

## 环境配置

创建环境：

```bash
conda create -n power python=3.10
conda activate power
```

安装依赖：

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn

pip install torch torchvision torchaudio

pip install holidays
```

---

## 模型训练

### LSTM

短期预测

```bash
python src/train_lstm.py \
    --output_len 90
```

长期预测

```bash
python src/train_lstm.py \
    --full_data_path data/processed/daily_power_weather.csv \
    --output_len 365
```

---

### Transformer

短期预测

```bash
python src/train_transformer.py \
    --output_len 90
```

长期预测

```bash
python src/train_transformer.py \
    --full_data_path data/processed/daily_power_weather.csv \
    --output_len 365
```

---

### TCN

短期预测

```bash
python src/train_tcn.py \
    --output_len 90
```

长期预测

```bash
python src/train_tcn.py \
    --full_data_path data/processed/daily_power_weather.csv \
    --output_len 365
```

---

## 评价指标

### 均方误差（MSE）

```text
MSE = mean((y_true - y_pred)^2)
```

用于衡量预测误差平方的平均值。

---

### 平均绝对误差（MAE）

```text
MAE = mean(|y_true - y_pred|)
```

用于衡量预测误差绝对值的平均水平。

---

## 实验设置

每个模型采用：

```text
随机种子：
1
2
3
4
5
```

训练流程：

```text
训练
 ↓
验证
 ↓
保存最佳模型
 ↓
统计MSE与MAE
 ↓
计算均值与标准差
```

---

## 结果展示

结果图片保存在：

```text
figures/
```

例如：

```text
lstm_output90_seed1.png
transformer_output90_seed1.png
tcn_output90_seed1.png
```

模型权重保存在：

```text
checkpoints/
```

例如：

```text
lstm_output90_seed1.pth
transformer_output90_seed1.pth
tcn_output90_seed1.pth
```

---

### 实验结果

#### 短期预测（90→90）

| 模型          |           MSE |        MAE |
| ----------- | ------------: | ---------: |
| LSTM        |     172096.50 |     306.53 |
| Transformer |     133936.18 |     278.11 |
| TCN         | **120414.88** | **267.51** |

#### 长期预测（90→365）

| 模型          |           MSE |        MAE |
| ----------- | ------------: | ---------: |
| LSTM        | **135343.72** | **281.02** |
| Transformer |     146505.45 |     292.63 |
| TCN         |     140994.39 |     288.09 |

---

## 结论

本项目比较了 LSTM、Transformer 和 TCN 三种深度学习模型在家庭电力消耗预测任务中的表现。

实验结果表明：

* LSTM 能够学习基本时间依赖关系，在长期预测任务中表现较好；
* Transformer 在目前规模的数据集中没有优势；
* TCN 在小规模时间序列数据集上具有较好的稳定性和预测性能。

未来可进一步研究：

* PatchTST
* Informer
* Autoformer
* FEDformer
* TimeXer
* 多模型集成预测
* 概率预测模型

---
## 作者

孟焕然

机器学习课程项目

家庭电力消耗预测（Household Power Consumption Forecasting）


