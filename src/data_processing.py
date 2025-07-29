# src/data_processing.py

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime

WEATHER_CACHE_PATH = 'data/weather_cache.csv'
DATA_PATH = 'data/data.xlsx'

def load_raw(path=DATA_PATH):
    df = pd.read_excel(path, parse_dates=['time_15min'])
    df = df.rename(columns={'time_15min': 'time'})
    df = df.set_index('time')
    df = df.sort_index()
    return df

def mark_holidays(df):
    """手动添加2024年中国主要法定节假日"""
    holiday_dates = [
        '2024-01-01', '2024-02-10', '2024-02-11', '2024-02-12', '2024-02-13', '2024-02-14', '2024-02-15', '2024-02-16',
        '2024-04-04', '2024-04-05', '2024-04-06',
        '2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-05-05',
        '2024-06-10', '2024-09-15', '2024-09-16', '2024-09-17',
        '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-05', '2024-10-06', '2024-10-07'
    ]
    holiday_dates = pd.to_datetime(holiday_dates)
    df['is_holiday'] = df.index.normalize().isin(holiday_dates).astype(int)
    return df

def fetch_weather():
    if os.path.exists(WEATHER_CACHE_PATH):
        print(f"[Weather] ✅ 已加载缓存天气数据： {WEATHER_CACHE_PATH}")
        df_weather = pd.read_csv(WEATHER_CACHE_PATH, parse_dates=['time'], index_col='time')
        return df_weather

    print("[Weather] 🔄 正在调用 Open-Meteo API 下载天气数据（2024-01-01 到 2024-12-31）")
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=30.2741&longitude=120.1551"
        "&start_date=2024-01-01&end_date=2024-12-31"
        "&hourly=temperature_2m,relative_humidity_2m,weathercode"
        "&timezone=Asia%2FShanghai"
    )
    print(f"[Weather] 请求 URL:\n{url}")
    response = requests.get(url)
    data = response.json()

    try:
        df_weather = pd.DataFrame({
            'time': data['hourly']['time'],
            'temperature_2m': data['hourly']['temperature_2m'],
            'relative_humidity_2m': data['hourly']['relative_humidity_2m'],
            'weathercode': data['hourly']['weathercode']
        })
        df_weather['time'] = pd.to_datetime(df_weather['time'])
        df_weather = df_weather.set_index('time').sort_index()
        df_weather.to_csv(WEATHER_CACHE_PATH)
        print(f"[Weather] ✅ 天气数据已保存至缓存：{WEATHER_CACHE_PATH}")
        return df_weather
    except KeyError:
        print("❌ 无法解析天气数据，请检查 API 响应格式是否变更")
        return pd.DataFrame()

def process_and_save():
    df_load = load_raw()
    df_weather = fetch_weather()

    # 合并天气与负荷数据
    df_all = pd.merge(df_load, df_weather, how='left', left_index=True, right_index=True)

    # ✅ 重命名负荷列
    df_all = df_all.rename(columns={'val': 'load'})

    # 添加节假日标识
    df_all = mark_holidays(df_all)

    # ✅ 设置索引列名，供后续 parse_dates 使用
    df_all.index.name = 'time_15min'

    # 保存处理后的数据
    df_all.to_csv('data/processed_data.csv')
    print("[Data] ✅ 数据已处理并保存：data/processed_data.csv")