# src/feature_engineering.py

import pandas as pd
import numpy as np
from pathlib import Path


def load_processed(path='data/processed_data.csv'):
    """
    加载处理后的数据文件
    """
    df = pd.read_csv(path, parse_dates=['time_15min'], index_col='time_15min')
    return df


def make_sliding_samples(df, lags=[1, 2, 3, 4]):
    """
    基于滑动窗口方法构造滞后特征，并添加时间与天气特征
    """
    df_feat = df.copy()

    # 滞后负荷特征
    for lag in lags:
        df_feat[f'lag_{lag}'] = df_feat['load'].shift(lag)

    # 时间特征
    df_feat['hour'] = df_feat.index.hour
    df_feat['minute'] = df_feat.index.minute
    df_feat['weekday'] = df_feat.index.weekday
    df_feat['is_weekend'] = df_feat['weekday'].isin([5, 6]).astype(int)

    # 删除缺失
    df_feat.dropna(inplace=True)

    # 特征与目标
    X = df_feat.drop(columns=['load'])
    y = df_feat['load']

    return X, y


def train_test_split_time_series(X, y, train_ratio=0.7):
    """
    时间序列拆分训练集与测试集
    """
    split_index = int(len(X) * train_ratio)
    return (
        X.iloc[:split_index], X.iloc[split_index:],
        y.iloc[:split_index], y.iloc[split_index:]
    )


def save_samples(X, y, out_path='data/samples/samples.csv'):
    """
    保存样本到 CSV
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df = X.copy()
    df['target'] = y
    df.to_csv(out_path, index=True)
    print(f"[Feature] ✅ 样本数据保存至：{out_path}")