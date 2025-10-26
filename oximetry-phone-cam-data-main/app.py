from flask import Flask, request, jsonify, render_template
import os
import tempfile
from spo2_predictor import SpO2Predictor

app = Flask(__name__)
predictor = SpO2Predictor()
predictor.load_model("spo2_model.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # 保存上传的视频到临时文件
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, 'temp_video.mp4')
    video_file.save(video_path)
    
    try:
        # 进行预测
        predictions = predictor.predict_from_video(video_path)
        
        # 清理临时文件
        os.remove(video_path)
        os.rmdir(temp_dir)
        
        # 返回预测结果
        return jsonify({
            'success': True,
            'predictions': predictions.tolist(),
            'average_spo2': float(predictions.mean()),
        })
    
    except Exception as e:
        # 确保清理临时文件
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)