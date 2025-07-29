# SmartMallLoadForecast

🎯 **项目名称：** 基于 XGBoost 的商业建筑用电负荷预测系统  
📍 **案例背景：** 以东部某大型商场 2024 年全年的用电数据为研究对象

## 📘 项目简介

本项目旨在构建一个高效、实用的电力负荷预测系统，结合滑动窗口法、节假日与天气特征工程，以及 XGBoost 回归模型，实现对商业建筑的短期用电量预测。

**主要功能包括：**
- ⏱️ 数据预处理与节假日标记
- ☁️ 自动下载并缓存当地历史天气数据（温度、湿度、天气代码）
- 🧠 特征工程：滞后负荷、时间变量、天气、节假日特征
- 🧪 滚动训练模拟部署预测场景
- 📈 误差评估与多种图表可视化输出
- 📦 模型保存与二次加载

## 🗂️ 项目结构
````
SmartMallLoadForecast/
├── data/                  # 原始数据与处理结果
│   ├── data.xlsx          # 原始用电数据（每15分钟）
│   ├── weather\_cache.csv  # 缓存的天气数据
│   ├── processed\_data.csv # 合并后的特征数据
│   └── samples/           # 构造好的训练样本
│
├── src/                   # 核心源代码
│   ├── data\_processing.py      # 数据加载、节假日标记与天气下载
│   ├── feature\_engineering.py  # 特征工程模块
│   ├── model\_train.py          # 训练与评估模型
│   └── visualize.py            # 可视化模块
│
├── data/results/          # 输出图表与模型
│   ├── \*.png              # 图像输出
│   └── xgb\_model.joblib   # 训练好的模型
│
├── main.py                # 主程序入口
├── environment.yml        # Conda 环境配置
├── .gitignore             # 忽略规则（忽略 data/）
└── README.md              # 项目说明

````

---

## 🚀 快速开始

### 1️⃣ 创建 Conda 环境并安装依赖

```bash
conda env create -f environment.yml
conda activate SmartMallLoadForecast
````

### 2️⃣ 准备数据

将原始用电数据（包含 `time_15min` 和 `val` 列）放入 `data/data.xlsx`

### 3️⃣ 运行主程序

```bash
python main.py
```

运行结束后，将自动输出以下内容：

* 处理后数据：`data/processed_data.csv`
* 滞后样本数据：`data/samples/samples.csv`
* 模型与评估图：`data/results/*.png`

---

## 📊 预测效果展示

* MAE：平均绝对误差
* RMSE：均方根误差
* 图表包括：

  * 实际 vs 预测 折线图
  * 散点拟合图
  * 特征重要性条形图

---

## 📚 模型与方法说明

* **算法选择：** XGBoost 回归模型
* **特征构建：**

  * 滞后负荷值（滑动窗口）
  * 小时、星期、是否节假日
  * 天气：温度、湿度、天气代码
* **评估策略：** 滚动预测模拟实际场景，评估泛化能力

---

## 🔗 依赖说明（节选）

* Python >= 3.9
* pandas, numpy, matplotlib, seaborn
* xgboost, scikit-learn
* openpyxl（用于读取 Excel）
* requests（用于天气查询）

完整依赖请查看 [`environment.yml`](./environment.yml)

```