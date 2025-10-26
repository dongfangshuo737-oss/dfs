import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

def extract_rgb_from_video(video_path):
    """从视频中提取RGB值
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        rgb_array: Shape为(frames, 3)的RGB数组，30Hz
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 计算整个帧的平均RGB值
        r = np.mean(frame[:, :, 2])  # OpenCV使用BGR顺序
        g = np.mean(frame[:, :, 1])
        b = np.mean(frame[:, :, 0])
        
        frames.append([r, g, b])
    
    cap.release()
    return np.array(frames)

def process_rgb_features(rgb_array, window_size=30):
    """处理RGB特征，计算时间窗口统计特征
    
    Args:
        rgb_array: Shape为(frames, 3)的RGB数组
        window_size: 时间窗口大小（默认30帧=1秒）
        
    Returns:
        features: Shape为(n_windows, 12)的特征数组
    """
    n_frames = len(rgb_array)
    n_windows = n_frames // window_size
    
    features = []
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window_data = rgb_array[start_idx:end_idx]
        
        # 为每个通道计算4个统计特征
        window_features = []
        for channel in range(3):
            channel_data = window_data[:, channel]
            window_features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.max(channel_data),
                np.min(channel_data)
            ])
        
        features.append(window_features)
    
    return np.array(features)

class SpO2Predictor:
    def __init__(self):
        self.scaler = None
        self.model = None
        
    def load_model(self, model_path):
        """加载训练好的模型和标准化器"""
        loaded = joblib.load(model_path)
        self.scaler = loaded['scaler']
        self.model = loaded['model']
    
    def predict_from_video(self, video_path):
        """从视频预测SpO2值
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            predictions: SpO2预测值数组
        """
        # 提取RGB值
        rgb_array = extract_rgb_from_video(video_path)
        
        # 处理特征
        features = process_rgb_features(rgb_array)
        
        # 标准化特征
        features_scaled = self.scaler.transform(features)
        
        # 预测
        predictions = self.model.predict(features_scaled)
        
        return predictions
    
if __name__ == "__main__":
    # 测试代码
    video_path = "test_video.mp4"
    predictor = SpO2Predictor()
    predictor.load_model("spo2_model.joblib")
    predictions = predictor.predict_from_video(video_path)
    print("SpO2 Predictions:", predictions)