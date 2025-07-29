# SmartMallLoadForecast

### 基于 XGBoost 的商业建筑用电负荷预测（以某大型商场为例）

---

## 一、项目概述

本项目旨在使用滑动窗口法和 XGBoost 模型，对某大型商场 2024 年全年 15 分钟级别用电数据进行短期预测（未来 15 分钟负荷）。可结合天气变量（温度）显著提升模型性能。通过滚动验证方式评估模型性能，并自动生成预测结果可视化图表。

---

## 二、项目结构

```

SmartMallLoadForecast/
│
├── environment.yml
├── README.md
├── data/                  # 被 .gitignore 忽略
│    └── your_file.csv
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_train.py
│   └── visualize.py
└── .gitignore

````

---

## 三、环境与依赖安装

确保使用 Python 3.8+ 或兼容环境（支持 macOS M1、Windows 系统）。

使用以下命令安装所需依赖：

```bash
pip install -r requirements.txt
````

`requirements.txt` 示例内容如下：

```
pandas
numpy
xgboost
scikit-learn
matplotlib
requests
```

---

## 四、数据准备格式说明

请将负荷数据存放于 `data/your_file.csv`，格式如下：

| time\_15min      | val   |
| ---------------- | ----- |
| 2024-01-01 00:00 | 78.5  |
| 2024-01-01 00:15 | 71.53 |
| 2024-01-01 00:30 | 78.26 |
| ...              | ...   |

* `time_15min` 列须为 `%Y-%m-%d %H:%M` 格式。
* `val` 列为该时刻的实际用电量（单位 kWh）。

---

## 五、运行步骤

1. 把数据 CSV 放入 `data/` 目录，并命名为 `your_file.csv`。
2. 在终端中进入 `load_forecast/` 项目目录。
3. 运行数据处理流程（包含天气、节假日特征的整合）：

```bash
python src/data_processing.py
```

4. 构造滑动窗口特征并训练模型：

```bash
python src/feature_engineering.py
python src/model_train.py
```

5. 结果可视化与性能评估：

```bash
python src/visualize.py
```

---

## 六、关键功能说明

* `data_processing.py`：读取原始负荷 CSV，自动标注 2024 年中国法定节假日，调用 Open‑Meteo API 获取杭州天气特征，并合并至数据集中。
* `feature_engineering.py`：生成滞后特征（如过去 4 个时间步历史负荷）、时间变量（小时、星期、是否节假日等）并输出特征矩阵。
* `model_train.py`：使用 XGBoost 进行滚动训练与验证，生成预测结果，同时输出 RMSE、MAE 等评估指标并保存训练好的模型。
* `visualize.py`：绘制预测 vs 实际对比图以及特征重要性图表，生成用于论文展示的图形输出。

---

## 七、样例结果展示（论文用示例）

* **指标结果**：

  * RMSE: 12.34 kWh
  * MAE: 8.56 kWh

* **图表说明**：

  * `fig_load_vs_pred.png`：预测曲线与实际负荷对比（线图）
  * `fig_feature_importance.png`：特征重要性柱状图

---

## 八、扩展建议

* 如有客流量、室外温度传感器、促销活动信息，可作为额外特征纳入模型。
* 若需预测未来 1 小时平均负荷，可将滑窗和输出目标调整为过去 4 个时段平均值预测未来 4 个时段平均。
* 可对比 RandomForest、LightGBM、LSTM 等模型性能，增强论文深度。