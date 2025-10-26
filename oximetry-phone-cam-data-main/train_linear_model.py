import h5py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

# 读取数据
def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        return {key: f[key][:] for key in f.keys()}

# 重采样函数：从30Hz降到1Hz
def downsample(data, factor=30):
    return data[::factor]

# 评估模型性能
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

# 主函数
def main():
    # 加载数据
    data = load_data('data/preprocessed/all_uw_data.h5')
    
    # 分别处理左右手数据
    hands = ['Left', 'Right']
    results = {}
    
    for hand in hands:
        print(f"\n处理{hand}手数据:")
        
        # 获取RGB数据 (30Hz)
        rgb_data = np.stack([
            data[f'{hand}_red'],
            data[f'{hand}_green'],
            data[f'{hand}_blue']
        ], axis=1)
        
        # 获取SpO2数据 (1Hz)
        spo2_data = data['reference_spo2']
        
        # 将RGB数据降采样到1Hz
        rgb_data_1hz = downsample(rgb_data, 30)
        
        # 确保数据长度匹配
        min_len = min(len(rgb_data_1hz), len(spo2_data))
        X = rgb_data_1hz[:min_len]
        y = spo2_data[:min_len]
        
        # 训练线性模型
        model = LinearRegression()
        model.fit(X, y)
        
        # 预测和评估
        y_pred = model.predict(X)
        mae, mse, r2 = evaluate_model(y, y_pred)
        
        # 保存结果
        results[hand] = {
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'mae': mae,
            'mse': mse,
            'r2': r2
        }
        
        print(f"模型系数 (R,G,B):", model.coef_)
        print(f"截距:", model.intercept_)
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"决定系数 (R²): {r2:.4f}")

if __name__ == "__main__":
    main()