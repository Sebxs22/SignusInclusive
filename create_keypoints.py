import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Desactivar advertencias de OneDNN

import warnings
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from helpers import get_keypoints, insert_keypoints_sequence
from constants import DATA_PATH, FRAME_ACTIONS_PATH, ROOT_PATH

def create_keypoints(frames_path, save_path):
    '''
    ### CREAR KEYPOINTS PARA UNA PALABRA
    Recorre la carpeta de frames de la palabra y guarda sus keypoints en `save_path`.
    
    Parámetros:
    - frames_path: Ruta de la carpeta que contiene los frames de la palabra.
    - save_path: Ruta donde se guardará el archivo .h5 con los keypoints.
    '''
    # Verificar que la carpeta de frames existe
    if not os.path.exists(frames_path):
        raise FileNotFoundError(f"La carpeta de frames no existe: {frames_path}")

    # Verificar que hay archivos en la carpeta
    frames = [f for f in os.listdir(frames_path) if f.endswith(".jpg")]
    if not frames:
        raise ValueError(f"No se encontraron frames en la carpeta: {frames_path}")

    # DataFrame para almacenar los keypoints
    data = pd.DataFrame([])

    # Procesar cada frame con MediaPipe Holistic
    with Holistic() as model_holistic:
        for n_sample, sample_name in enumerate(frames, 1):
            sample_path = os.path.join(frames_path, sample_name)
            keypoints_sequence = get_keypoints(model_holistic, sample_path)
            data = insert_keypoints_sequence(data, n_sample, keypoints_sequence)

    # Guardar los keypoints en un archivo .h5
    data.to_hdf(save_path, key="data", mode="w")
    print(f"Keypoints guardados en {save_path}.")

if __name__ == "__main__":
    words_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH)
    
    # Ignorar advertencias
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # GENERAR LOS KEYPOINTS DE TODAS LAS PALABRAS
        for word_name in os.listdir(words_path):
            word_path = os.path.join(words_path, word_name)
            hdf_path = os.path.join(DATA_PATH, f"{word_name}.h5")
            
            print(f'Creando keypoints de "{word_name}"...')
            try:
                create_keypoints(word_path, hdf_path)
                print(f"Keypoints creados para '{word_name}'!")
            except Exception as e:
                print(f"Error al crear keypoints para '{word_name}': {e}")