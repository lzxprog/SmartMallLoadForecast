# src/model_train.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# 设置中文显示
plt.rcParams['font.family'] = 'Microsoft YaHei'  # 或者 'SimHei'，根据系统字体
plt.rcParams['axes.unicode_minus'] = False


def load_samples(path='data/samples/samples.csv'):
    """
    读取训练样本数据
    """
    df = pd.read_csv(path, parse_dates=['time_15min'], index_col='time_15min')
    return df


def walk_forward_train_predict(df, train_ratio=0.7):
    """
    滚动训练预测：模拟实际部署中的预测
    """
    features = df.columns.drop('target')
    split_index = int(len(df) * train_ratio)

    preds = []
    trues = []

    for i in range(split_index, len(df)):
        train_df = df.iloc[:i]
        test_row = df.iloc[[i]]

        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        model.fit(train_df[features], train_df['target'])
        pred = model.predict(test_row[features])[0]

        preds.append(pred)
        trues.append(test_row['target'].values[0])

    return trues, preds, model


def evaluate(y_true, y_pred):
    """
    模型评估指标
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[评估] ✅ MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    return mae, rmse


def plot_results(y_true, y_pred, output_dir='data/results'):
    """
    可视化预测值与实际值
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='实际值')
    plt.plot(y_pred, label='预测值')
    plt.title('实际值 vs 预测值')
    plt.xlabel('时间步')
    plt.ylabel('负荷（kW）')
    plt.legend()
    plt.tight_layout()
    save_path = f"{output_dir}/actual_vs_pred.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[图表] 📊 已保存预测对比图：{save_path}")


def plot_feature_importance(model, feature_names, output_dir='data/results'):
    """
    可视化特征重要性
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx])
    plt.gca().invert_yaxis()
    plt.title('特征重要性')
    plt.xlabel('重要性得分')
    plt.tight_layout()
    save_path = f"{output_dir}/feature_importance.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[图表] 📊 已保存特征重要性图：{save_path}")


def save_model(model, path='data/results/xgb_model.joblib'):
    """
    保存模型
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[模型] 💾 模型已保存至：{path}")


def load_model(path='data/results/xgb_model.joblib'):
    """
    加载模型
    """
    return joblib.load(path)