import h5py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from scipy import stats

def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        # dataset shape: (6, 6, 33660) - 6个受试者，每个受试者6个通道（左右手各3个RGB），每个通道33660个时间点
        # groundtruth shape: (6, 5, 1122) - 6个受试者，5种测量值，每种1122个时间点
        
        # 获取左手RGB数据 (30Hz) - 索引0,1,2对应左手
        rgb_data = f['dataset'][:, 0:3, :]  # 获取所有受试者的左手RGB数据
        
        # 获取所有SpO2数据列
        spo2_data = f['groundtruth'][:, :, :]  # 获取所有SpO2测量值
        
        # 排除受试者1的数据
        mask = np.ones(len(rgb_data), dtype=bool)
        mask[1] = False
        
        return rgb_data[mask], spo2_data[mask]

def process_spo2_groundtruth(spo2_data):
    """
    处理SpO2数据以创建更可靠的ground truth
    
    Args:
        spo2_data: Shape为(n_subjects, n_measurements, n_timestamps)的SpO2数据
    
    Returns:
        processed_spo2: Shape为(n_subjects, n_timestamps)的处理后的SpO2数据
    """
    n_subjects, n_measurements, n_timestamps = spo2_data.shape
    processed_spo2 = np.zeros((n_subjects, n_timestamps))
    
    for subject in range(n_subjects):
        for t in range(n_timestamps):
            measurements = spo2_data[subject, :, t]
            
            # 移除异常值
            # 使用Z-score方法识别异常值
            z_scores = np.abs(stats.zscore(measurements))
            valid_mask = z_scores < 2  # 使用2个标准差作为阈值
            valid_measurements = measurements[valid_mask]
            
            if len(valid_measurements) > 0:
                # 计算每个有效测量值的权重
                # 使用与中位数的接近程度作为权重
                median_value = np.median(valid_measurements)
                weights = 1 / (np.abs(valid_measurements - median_value) + 1e-6)
                
                # 计算加权平均
                processed_spo2[subject, t] = np.sum(valid_measurements * weights) / np.sum(weights)
            else:
                # 如果所有值都被视为异常值，使用原始中位数
                processed_spo2[subject, t] = np.median(measurements)
    
    return processed_spo2

def prepare_features(rgb_data, window_size=30):
    """准备RGB特征"""
    n_subjects, n_channels, n_timestamps = rgb_data.shape
    n_windows = n_timestamps // window_size
    
    # 初始化特征数组
    features = np.zeros((n_subjects, n_windows, n_channels * 4))  # 4个统计特征：均值、标准差、最大值、最小值
    
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

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

def main():
    print("使用加权平均和异常值移除处理SpO2数据:")
    
    # 加载数据
    rgb_data, spo2_data = load_data('data/preprocessed/all_uw_data.h5')
    
    # 处理SpO2 ground truth
    processed_spo2 = process_spo2_groundtruth(spo2_data)
    
    # 准备特征
    features = prepare_features(rgb_data)
    
    # 重塑数据以适应训练
    n_subjects, n_windows, n_features = features.shape
    X = features.reshape(-1, n_features)
    y = processed_spo2.reshape(-1)
    
    # 创建受试者索引用于交叉验证
    subject_indices = np.repeat(np.arange(n_subjects), n_windows)
    
    # 存储交叉验证结果
    cv_results = {
        'mae': [], 'mse': [], 'r2': [],
        'coefficients': [], 'intercepts': []
    }
    
    # 使用RobustScaler进行特征标准化
    scaler = RobustScaler()
    
    # 留一交叉验证
    unique_subjects = np.unique(subject_indices)
    for test_subject in unique_subjects:
        # 划分训练集和测试集
        train_mask = subject_indices != test_subject
        test_mask = subject_indices == test_subject
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # 标准化特征
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # 预测和评估
        y_pred = model.predict(X_test_scaled)
        mae, mse, r2 = evaluate_model(y_test, y_pred)
        
        # 存储结果
        cv_results['mae'].append(mae)
        cv_results['mse'].append(mse)
        cv_results['r2'].append(r2)
        cv_results['coefficients'].append(model.coef_)
        cv_results['intercepts'].append(model.intercept_)
    
    # 输出结果
    print("\n交叉验证结果 (均值 ± 标准差):")
    print(f"MAE: {np.mean(cv_results['mae']):.4f} ± {np.std(cv_results['mae']):.4f}")
    print(f"MSE: {np.mean(cv_results['mse']):.4f} ± {np.std(cv_results['mse']):.4f}")
    print(f"R²: {np.mean(cv_results['r2']):.4f} ± {np.std(cv_results['r2']):.4f}")
    
    # 特征重要性分析
    mean_coef = np.mean(cv_results['coefficients'], axis=0)
    std_coef = np.std(cv_results['coefficients'], axis=0)
    
    # 特征名称
    channels = ['R', 'G', 'B']
    stats = ['Mean', 'Std', 'Max', 'Min']
    feature_names = [f"{ch}_{stat}" for ch in channels for stat in stats]
    
    print("\n特征重要性 (系数 ± 标准差):")
    for name, coef, std in zip(feature_names, mean_coef, std_coef):
        print(f"{name}: {coef:.4f} ± {std:.4f}")
    
    # 计算各特征的相对重要性
    importance = np.abs(mean_coef) / np.sum(np.abs(mean_coef)) * 100
    
    print("\n特征相对重要性 (%):")
    for name, imp in zip(feature_names, importance):
        print(f"{name}: {imp:.2f}%")
    
    # 输出每个受试者的结果
    print("\n各受试者个体结果:")
    for i, subject in enumerate(unique_subjects):
        print(f"\n受试者 {subject}:")
        print(f"MAE: {cv_results['mae'][i]:.4f}")
        print(f"MSE: {cv_results['mse'][i]:.4f}")
        print(f"R²: {cv_results['r2'][i]:.4f}")

if __name__ == "__main__":
    main()