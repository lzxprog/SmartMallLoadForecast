# src/visualize.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

# 设置中文字体（根据系统调整）
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False


def load_results(samples_path='data/samples/samples.csv', model_path='data/results/xgb_final_model.joblib'):
    df = pd.read_csv(samples_path, parse_dates=['time_15min'], index_col='time_15min')
    model = joblib.load(model_path)
    return df, model


def predict_all(model, df):
    X = df.drop(columns=['target'])
    y = df['target']
    y_pred = model.predict(X)
    return y, y_pred, X.columns.tolist()


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[Metrics] 📐 全量预测 MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    return mae, rmse


def plot_load_vs_pred(y_true, y_pred, time_index, output_dir='data/results'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    plt.figure(figsize=(14, 5))
    plt.plot(time_index, y_true, label='实际值', linewidth=1.5)
    plt.plot(time_index, y_pred, label='预测值', linewidth=1.5)
    plt.title(f'实际负荷 vs 预测负荷（MAE={mae:.2f}, RMSE={rmse:.2f}）')
    plt.xlabel('时间')
    plt.ylabel('负荷 (kW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = f"{output_dir}/actual_vs_pred_full_with_metrics.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] 📈 折线图已保存：{out_path}")


def plot_scatter_actual_pred(y_true, y_pred, output_dir='data/results'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    sns.regplot(x=y_true, y=y_pred, scatter_kws={"s": 10, "alpha": 0.6}, line_kws={"color": "red"})
    plt.xlabel("实际负荷")
    plt.ylabel("预测负荷")
    plt.title("散点图：实际值 vs 预测值")
    plt.grid(True)
    plt.tight_layout()
    out_path = f"{output_dir}/scatter_actual_pred.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] 🟢 散点图已保存：{out_path}")


def plot_residual_distribution(y_true, y_pred, output_dir='data/results'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=40, color='steelblue')
    plt.title("预测残差分布图")
    plt.xlabel("残差（实际值 - 预测值）")
    plt.ylabel("频数")
    plt.grid(True)
    plt.tight_layout()
    out_path = f"{output_dir}/residual_distribution.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] 📉 残差分布图已保存：{out_path}")


FEATURE_NAME_MAP = {
    'lag_1': '滞后1期',
    'lag_2': '滞后2期',
    'lag_3': '滞后3期',
    'lag_4': '滞后4期',
    'hour': '小时',
    'minute': '分钟',
    'weekday': '星期几',
    'is_weekend': '是否周末',
    'is_holiday': '是否节假日',
    'temperature_2m': '气温',
    'relative_humidity_2m': '湿度',
    'weathercode': '天气代码'
}


def plot_feature_importance(model, feature_names, output_dir='data/results'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    feature_names_cn = [FEATURE_NAME_MAP.get(f, f) for f in feature_names]
    feature_names_sorted = np.array(feature_names_cn)[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names_sorted, importance[sorted_idx], color='skyblue')
    plt.gca().invert_yaxis()
    plt.title('特征重要性')
    plt.xlabel('重要性得分')
    plt.tight_layout()
    out_path = f"{output_dir}/feature_importance_visual.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] 📊 特征重要性图已保存：{out_path}")