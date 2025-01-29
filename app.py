from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import threading
from capture_samples import capture_samples
from create_keypoints import create_keypoints
from training_model import training_model
from evaluate_model import evaluate_model
from keras.models import load_model

app = Flask(__name__)
CORS(app)  # Permitir CORS para todas las rutas

# Función para capturar muestras y generar keypoints
def capture_and_generate_keypoints(word_name):
    root = os.getcwd()
    words_path = os.path.join(root, "frame_actions")
    data_path = os.path.join(root, "data")

    # Capturar muestras para la palabra ingresada
    word_path = os.path.join(words_path, word_name)
    capture_samples(word_path)

    # Generar los keypoints de todas las palabras
    for word_name in os.listdir(words_path):
        word_path = os.path.join(words_path, word_name)
        hdf_path = os.path.join(data_path, f"{word_name}.h5")
        print(f'Creando keypoints de "{word_name}"...')
        create_keypoints(word_path, hdf_path)

# Función para entrenar el modelo
def train_model():
    try:
        root = os.getcwd()
        data_path = os.path.join(root, "data")
        save_path = os.path.join(root, "models")
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, "trained_model.h5")

        print("Iniciando entrenamiento del modelo...")
        training_model(data_path, model_path)
        print("Modelo entrenado exitosamente.")
    except Exception as e:
        print(f"Error al entrenar el modelo: {e}")

# Función para evaluar el modelo
def evaluate_model_thread():
    try:
        model_path = os.path.join("models", "trained_model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El modelo no se encontró en la ruta: {model_path}")

        lstm_model = load_model(model_path)  # Cargar el modelo
        print("Iniciando evaluación del modelo...")
        evaluate_model(lstm_model)  # Evaluar el modelo
        print("Evaluación del modelo finalizada.")
    except Exception as e:
        print(f"Error durante la evaluación del modelo: {e}")

# Endpoint para guardar la palabra y procesarla
@app.route('/save_word', methods=['POST'])
def save_word():
    data = request.get_json()
    word = data.get("word")
    if word:
        # Ejecutar el proceso de captura y generación de keypoints en un hilo separado
        thread = threading.Thread(target=capture_and_generate_keypoints, args=(word,))
        thread.start()
        return jsonify({"message": f"Procesando la palabra '{word}' en segundo plano."})
    else:
        return jsonify({"error": "No se proporcionó una palabra."}), 400

# Endpoint para entrenar el modelo
@app.route('/train', methods=['POST'])
def train():
    try:
        # Ejecutar el proceso de entrenamiento en un hilo separado
        thread = threading.Thread(target=train_model)
        thread.start()
        return jsonify({"message": "El modelo se está entrenando en segundo plano."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint para evaluar el modelo
@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        # Ejecutar la evaluación en un hilo separado
        thread = threading.Thread(target=evaluate_model_thread)
        thread.start()
        return jsonify({"message": "La evaluación del modelo se está ejecutando en segundo plano."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ruta de inicio
@app.route('/')
def index():
    return jsonify({"message": "API de Flask funcionando correctamente."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Hacer el servidor accesible en todas las interfaces
