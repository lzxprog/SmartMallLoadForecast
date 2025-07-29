# src/visualize.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

# 设置中文字体（确保系统支持，如 Windows 使用 Microsoft YaHei，Mac 可用 Heiti TC）
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_results(samples_path='data/samples/samples.csv', model_path='data/results/xgb_final_model.joblib'):
    """
    加载数据与训练好的模型
    """
    df = pd.read_csv(samples_path, parse_dates=['time_15min'], index_col='time_15min')
    model = joblib.load(model_path)
    return df, model


def predict_all(model, df):
    """
    使用模型对完整数据集进行预测
    """
    X = df.drop(columns=['target'])
    y = df['target']
    y_pred = model.predict(X)
    return y, y_pred, X.columns.tolist()


def compute_metrics(y_true, y_pred):
    """
    计算 MAE 和 RMSE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[Metrics] 📐 全量预测 MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    return mae, rmse


def plot_load_vs_pred(y_true, y_pred, time_index, output_dir='data/results'):
    """
    实际负荷 vs 预测负荷 折线图
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(time_index, y_true, label='实际值', linewidth=1.5)
    plt.plot(time_index, y_pred, label='预测值', linewidth=1.5)
    plt.title('实际负荷 vs 预测负荷')
    plt.xlabel('时间')
    plt.ylabel('负荷 (kW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = f"{output_dir}/actual_vs_pred_full.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] 📈 折线图已保存：{out_path}")


def plot_scatter_actual_pred(y_true, y_pred, output_dir='data/results'):
    """
    散点图：实际 vs 预测
    """
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


def plot_feature_importance(model, feature_names, output_dir='data/results'):
    """
    特征重要性图（条形图）
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx])
    plt.gca().invert_yaxis()
    plt.title('特征重要性')
    plt.tight_layout()
    out_path = f"{output_dir}/feature_importance_visual.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] 📊 特征重要性图已保存：{out_path}")