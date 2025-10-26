import h5py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
import joblib

def load_training_data(file_path):
    """加载训练数据
    
    Args:
        file_path: HDF5文件路径
        
    Returns:
        rgb_data: RGB数据
        spo2_data: SpO2数据
    """
    with h5py.File(file_path, 'r') as f:
        # 获取左手RGB数据 (30Hz)
        rgb_data = f['dataset'][:, 0:3, :]
        
        # 获取SpO2数据 (1Hz)
        spo2_data = f['groundtruth'][:, 0, :]
        
        # 排除受试者1
        mask = np.ones(len(rgb_data), dtype=bool)
        mask[1] = False
        
        return rgb_data[mask], spo2_data[mask]

def process_features(rgb_data, window_size=30):
    """处理RGB特征"""
    n_subjects, n_channels, n_timestamps = rgb_data.shape
    n_windows = n_timestamps // window_size
    
    # 初始化特征数组 (12个特征：每个通道的均值、标准差、最大值、最小值)
    features = np.zeros((n_subjects, n_windows, n_channels * 4))
    
    for subject in range(n_subjects):
        for w in range(n_windows):
            start_idx = w * window_size
            end_idx = start_idx + window_size
            
            for c in range(n_channels):
                window_data = rgb_data[subject, c, start_idx:end_idx]
                feature_idx = c * 4
                
                # 计算统计特征
                features[subject, w, feature_idx] = np.mean(window_data)
                features[subject, w, feature_idx + 1] = np.std(window_data)
                features[subject, w, feature_idx + 2] = np.max(window_data)
                features[subject, w, feature_idx + 3] = np.min(window_data)
    
    return features

def train_model():
    """训练SpO2预测模型"""
    # 加载数据
    rgb_data, spo2_data = load_training_data('data/preprocessed/all_uw_data.h5')
    
    # 处理特征
    features = process_features(rgb_data)
    
    # 重塑数据
    n_subjects, n_windows, n_features = features.shape
    X = features.reshape(-1, n_features)
    
    # 对SpO2数据进行下采样，使其与特征数据匹配
    y = []
    samples_per_second = 30  # RGB数据的采样率
    for subject in range(len(spo2_data)):
        spo2_windows = spo2_data[subject][:n_windows]  # 确保长度匹配
        y.extend(spo2_windows)
    y = np.array(y)
    
    print(f"特征形状: {X.shape}")
    print(f"标签形状: {y.shape}")
    
    # 标准化特征
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练模型
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # 保存模型和标准化器
    model_data = {
        'model': model,
        'scaler': scaler
    }
    joblib.dump(model_data, 'spo2_model.joblib')
    
    return model, scaler

if __name__ == "__main__":
    print("训练SpO2预测模型...")
    model, scaler = train_model()
    print("模型训练完成并保存到 spo2_model.joblib")