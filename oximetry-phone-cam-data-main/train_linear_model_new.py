import h5py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

# 读取数据
def load_and_prepare_data(file_path, hand_idx):
    with h5py.File(file_path, 'r') as f:
        # dataset shape: (6, 6, 33660) - 6个受试者，每个受试者6个通道（左右手各3个RGB），每个通道33660个时间点
        # groundtruth shape: (6, 5, 1122) - 6个受试者，5种测量值，每种1122个时间点
        
        # 获取RGB数据 (30Hz)
        # hand_idx: 0 for left hand (0,1,2), 1 for right hand (3,4,5)
        start_idx = hand_idx * 3
        rgb_data = f['dataset'][:, start_idx:start_idx+3, :]  # 获取所有受试者的指定手的RGB数据
        
        # 获取SpO2数据 (1Hz) - 假设在groundtruth中的第一列
        spo2_data = f['groundtruth'][:, 0, :]  # 获取所有受试者的SpO2数据
        
        return rgb_data, spo2_data

# 重采样函数：从30Hz降到1Hz
def downsample(data, factor=30):
    if len(data.shape) == 3:  # 对于RGB数据
        return data[:, :, ::factor]
    else:  # 对于SpO2数据
        return data

def prepare_features_and_target(rgb_data, spo2_data):
    # 将所有受试者的数据组合在一起
    X_list = []
    y_list = []
    
    n_subjects = rgb_data.shape[0]
    for i in range(n_subjects):
        rgb = rgb_data[i]  # (3, timestamps)
        spo2 = spo2_data[i]  # (timestamps,)
        
        # 将RGB数据转置为 (timestamps, 3)
        rgb = rgb.T
        
        # 确保长度匹配
        min_len = min(len(rgb), len(spo2))
        X_list.append(rgb[:min_len])
        y_list.append(spo2[:min_len])
    
    return np.vstack(X_list), np.concatenate(y_list)

# 评估模型性能
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

# 主函数
def main():
    # 为左右手分别训练模型
    hands = ['Left', 'Right']
    hand_indices = [0, 1]  # 0 for left hand, 1 for right hand
    results = {}
    
    for hand, hand_idx in zip(hands, hand_indices):
        print(f"\n处理{hand}手数据:")
        
        # 加载数据
        rgb_data, spo2_data = load_and_prepare_data('data/preprocessed/all_uw_data.h5', hand_idx)
        
        # 降采样RGB数据从30Hz到1Hz
        rgb_data_1hz = downsample(rgb_data)
        
        # 准备特征和目标变量
        X, y = prepare_features_and_target(rgb_data_1hz, spo2_data)
        
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