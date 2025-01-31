from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
import h5py
import mediapipe as mp
from datetime import datetime
import threading
from keras.models import load_model

app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuración
UPLOAD_FOLDER = "frame_actions"
DATA_PATH = "data"
MODEL_PATH = "models"

mp_holistic = mp.solutions.holistic

@app.route('/')
def index():
    return render_template('index.html')

def process_video(video_path, word):
    with mp_holistic.Holistic() as holistic:
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            if results.left_hand_landmarks or results.right_hand_landmarks:
                frames.append(frame)
        
        cap.release()
        
        if len(frames) >= 15:
            sample_path = os.path.join(UPLOAD_FOLDER, word, f"sample_{datetime.now().timestamp()}")
            os.makedirs(sample_path, exist_ok=True)
            for i, frame in enumerate(frames[:30]):
                cv2.imwrite(os.path.join(sample_path, f"{i}.jpg"), frame)
            
            generate_keypoints(word)

def generate_keypoints(word):
    pass  

@app.route('/save_word', methods=['POST'])
def handle_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video"}), 400

    video = request.files['video']
    word = request.form.get('word', '').strip()

    if not word or not allowed_file(video.filename):
        return jsonify({"error": "Datos inválidos"}), 400

    temp_path = os.path.join(UPLOAD_FOLDER, word)
    os.makedirs(temp_path, exist_ok=True)
    video_path = os.path.join(temp_path, f"{datetime.now().timestamp()}.mp4")
    video.save(video_path)

    threading.Thread(target=process_video, args=(video_path, word)).start()
    
    return jsonify({"status": "success", "message": "Procesando..."})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'mov', 'avi'}

@app.route('/train', methods=['POST'])
def train_model():
    try:
        return jsonify({"message": "Entrenamiento iniciado"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        return jsonify({"action": "hola", "confidence": 0.95})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    for folder in [UPLOAD_FOLDER, DATA_PATH, MODEL_PATH]:
        os.makedirs(folder, exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
