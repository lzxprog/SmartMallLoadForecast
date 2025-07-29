# @auth: Lizx
# @date: 2025-07-29

# main.py

import os
import pandas as pd
from src import data_processing, feature_engineering, model_train, visualize


def main():
    print("====== SmartMallLoadForecast Pipeline Starting ======")

    # 1. 数据处理：读取原始负荷数据，标注节假日与天气特征
    print("Step 1: Data processing ...")
    data_processing.process_and_save()

    # 2. 特征工程：构建滞后特征、时间与天气变量
    print("Step 2: Feature engineering ...")
    df = feature_engineering.load_processed(path='data/processed_data.csv')
    X, y = feature_engineering.make_sliding_samples(df, lags=[1, 2, 3, 4])
    X_train, X_test, y_train, y_test = feature_engineering.train_test_split_time_series(X, y, train_ratio=0.7)
    feature_engineering.save_samples(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

    # 3. 模型训练与滚动验证：XGBoost 逐步训练并保存模型及预测结果
    print("Step 3: Model training and validation ...")
    trues, preds, model = model_train.walk_forward_train_predict(
        pd.read_csv('data/samples/samples.csv', parse_dates=['time_15min'], index_col='time_15min'), train_ratio=0.7)
    mae, rmse = model_train.evaluate(trues, preds)
    model_train.plot_results(trues, preds, output_dir='data/results')
    feature_cols = model_train.load_samples('data/samples/samples.csv').columns.drop('target').tolist()
    model_train.plot_feature_importance(model, feature_cols, output_dir='data/results')
    model_train.save_model(model, 'data/results/xgb_final_model.joblib') if hasattr(model_train, 'save_model') else None

    # 4. 可视化模块：生成论文用图表并输出指标
    print("Step 4: Visualization ...")
    df_full, final_model = visualize.load_results(samples_path='data/samples/samples.csv',
                                                  model_path='data/results/xgb_final_model.joblib')
    y_true, y_pred, feat_cols = visualize.predict_all(final_model, df_full)
    mae2, rmse2 = visualize.compute_metrics(y_true, y_pred)
    print(f"Full-set performance: MAE={mae2:.3f}, RMSE={rmse2:.3f}")
    visualize.plot_load_vs_pred(y_true, y_pred, df_full.index, output_dir='data/results')
    visualize.plot_scatter_actual_pred(y_true, y_pred, output_dir='data/results')
    visualize.plot_feature_importance(final_model, feat_cols, output_dir='data/results')

    print("====== Pipeline Completed Successfully ======")


if __name__ == "__main__":
    main()