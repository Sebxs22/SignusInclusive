import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH

def capture_samples(path, margin_frame=2, min_cant_frames=5):
    create_folder(path)
    
    cant_sample_exist = len(os.listdir(path))
    count_sample = 0
    count_frame = 0
    frames = []
    capturing = False
    
    def save_frames(frames, output_folder):
        for num_frame, frame in enumerate(frames):
            frame_path = os.path.join(output_folder, f"{num_frame + 1}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            frame = video.read()[1] 
            image, results = mediapipe_detection(frame, holistic_model)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                capturing = not capturing
                if not capturing:
                    if len(frames) > min_cant_frames + margin_frame:
                        frames = frames[:-margin_frame]
                        output_folder = os.path.join(path, f"sample_{cant_sample_exist + count_sample + 1}")
                        create_folder(output_folder)
                        print(f"Guardando {len(frames)} frames en {output_folder}")
                        save_frames(frames, output_folder)
                        count_sample += 1
                    
                    frames = []
                    count_frame = 0
            
            if capturing:
                if there_hand(results):
                    count_frame += 1
                    if count_frame > margin_frame: 
                        cv2.putText(image, 'Capturando...', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                        frames.append(np.asarray(frame))
                else:
                    count_frame = 0
                    cv2.putText(image, 'Mano no detectada...', FONT_POS, FONT, FONT_SIZE, (0, 0, 255))
            else:
                cv2.putText(image, 'Listo para capturar... Presiona "s"', FONT_POS, FONT, FONT_SIZE, (0, 220, 100))

            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)
            if key == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Leer las palabras directamente de la carpeta frame_actions
    actions_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH)
    valid_words = [folder for folder in os.listdir(actions_path) if os.path.isdir(os.path.join(actions_path, folder))]
    
    # Preguntar al usuario por la palabra
    while True:
        word_name = input(f"¿Qué palabra deseas capturar? (opciones: {', '.join(valid_words)})\n")
        if word_name in valid_words:
            print(f"Palabra seleccionada: {word_name}")
            break
        else:
            print("Esa palabra no existe. Por favor, elige una palabra válida.")
    
    # Continuar con la captura de muestras
    word_path = os.path.join(actions_path, word_name)
    capture_samples(word_path)
