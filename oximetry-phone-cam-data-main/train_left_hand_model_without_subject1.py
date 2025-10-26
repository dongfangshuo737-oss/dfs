import h5py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 读取数据
def load_and_prepare_data(file_path):
    with h5py.File(file_path, 'r') as f:
        # dataset shape: (6, 6, 33660) - 6个受试者，每个受试者6个通道（左右手各3个RGB），每个通道33660个时间点
        # groundtruth shape: (6, 5, 1122) - 6个受试者，5种测量值，每种1122个时间点
        
        # 获取左手RGB数据 (30Hz) - 索引0,1,2对应左手
        rgb_data = f['dataset'][:, 0:3, :]  # 获取所有受试者的左手RGB数据
        
        # 获取SpO2数据 (1Hz)
        spo2_data = f['groundtruth'][:, 0, :]  # 获取所有受试者的SpO2数据
        
        # 排除受试者1的数据（索引为1）
        mask = np.ones(len(rgb_data), dtype=bool)
        mask[1] = False  # 排除索引为1的受试者
        
        return rgb_data[mask], spo2_data[mask]

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
    subject_indices = []  # 记录每个样本属于哪个受试者
    
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
        subject_indices.extend([i] * min_len)
    
    return np.vstack(X_list), np.concatenate(y_list), np.array(subject_indices)

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

# 主函数
def main():
    print("处理左手数据 (排除受试者1):")
    
    # 加载数据
    rgb_data, spo2_data = load_and_prepare_data('data/preprocessed/all_uw_data.h5')
    
    # 降采样RGB数据从30Hz到1Hz
    rgb_data_1hz = downsample(rgb_data)
    
    # 准备特征和目标变量
    X, y, subject_indices = prepare_features_and_target(rgb_data_1hz, spo2_data)
    
    # 初始化K折交叉验证（按受试者分组）
    n_splits = 5  # 现在只有5个受试者
    unique_subjects = np.unique(subject_indices)
    
    # 存储每折的结果
    cv_results = {
        'mae': [], 'mse': [], 'r2': [],
        'coefficients': [], 'intercepts': []
    }
    
    # 初始化标准化器
    scaler = StandardScaler()
    
    # 留一交叉验证（Leave-One-Subject-Out）
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
        
        # 预测
        y_pred = model.predict(X_test_scaled)
        
        # 评估
        mae, mse, r2 = evaluate_predictions(y_test, y_pred)
        
        # 存储结果
        cv_results['mae'].append(mae)
        cv_results['mse'].append(mse)
        cv_results['r2'].append(r2)
        cv_results['coefficients'].append(model.coef_)
        cv_results['intercepts'].append(model.intercept_)
    
    # 计算并输出平均结果
    print("\n交叉验证结果 (均值 ± 标准差):")
    print(f"MAE: {np.mean(cv_results['mae']):.4f} ± {np.std(cv_results['mae']):.4f}")
    print(f"MSE: {np.mean(cv_results['mse']):.4f} ± {np.std(cv_results['mse']):.4f}")
    print(f"R²: {np.mean(cv_results['r2']):.4f} ± {np.std(cv_results['r2']):.4f}")
    
    # 计算平均模型系数
    mean_coef = np.mean(cv_results['coefficients'], axis=0)
    std_coef = np.std(cv_results['coefficients'], axis=0)
    
    print("\n模型系数 (均值 ± 标准差):")
    print(f"R通道: {mean_coef[0]:.4f} ± {std_coef[0]:.4f}")
    print(f"G通道: {mean_coef[1]:.4f} ± {std_coef[1]:.4f}")
    print(f"B通道: {mean_coef[2]:.4f} ± {std_coef[2]:.4f}")
    
    # 计算通道重要性
    importance = np.abs(mean_coef) / np.sum(np.abs(mean_coef)) * 100
    print(f"\n各通道相对重要性:")
    print(f"R通道: {importance[0]:.2f}%")
    print(f"G通道: {importance[1]:.2f}%")
    print(f"B通道: {importance[2]:.2f}%")
    
    # 输出每个受试者的个体结果
    print("\n各受试者个体结果:")
    for i, subject in enumerate(unique_subjects):
        print(f"\n受试者 {subject}:")
        print(f"MAE: {cv_results['mae'][i]:.4f}")
        print(f"R²: {cv_results['r2'][i]:.4f}")

if __name__ == "__main__":
    main()