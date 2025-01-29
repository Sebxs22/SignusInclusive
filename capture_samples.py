import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH

def capture_samples(path, margin_frame=2, min_cant_frames=5):
    '''
    ### CAPTURA DE MUESTRAS PARA UNA PALABRA
    Recibe como parámetro la ubicación de guardado y guarda los frames.
    
    `path` ruta de la carpeta de la palabra.
    `margin_frame` cantidad de frames que se ignoran al comienzo y al final.
    `min_cant_frames` cantidad de frames mínimos para cada muestra.
    '''
    create_folder(path)
    
    cant_sample_exist = len(os.listdir(path))
    count_sample = 0
    count_frame = 0
    frames = []
    recording = False
    timer = 0  # Temporizador para mostrar en pantalla.

    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic_model)
            
            # Mostrar mensaje según el estado de grabación.
            if recording:
                timer += 1  # Incrementar temporizador durante grabación.
                cv2.putText(image, f'Grabando... Tiempo: {timer // 30}s', FONT_POS, FONT, FONT_SIZE, (255, 50, 0), 2)
                if there_hand(results):
                    count_frame += 1
                    if count_frame > margin_frame:
                        frames.append(np.asarray(frame))
                else:
                    cv2.putText(image, 'Mano no detectada', FONT_POS, FONT, FONT_SIZE, (0, 0, 255), 2)
            else:
                cv2.putText(image, 'Listo para grabar. Presione "s" para iniciar.', FONT_POS, FONT, FONT_SIZE, (0, 220, 100), 2)
            
            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)
            
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):  # Salir del programa.
                break
            elif key == ord('s'):  # Iniciar o detener grabación.
                if recording:
                    # Guardar las muestras si hay suficientes frames.
                    if len(frames) > min_cant_frames + margin_frame:
                        frames = frames[:-margin_frame]  # Eliminar frames al final.
                        output_folder = os.path.join(path, f"sample_{cant_sample_exist + count_sample + 1}")
                        create_folder(output_folder)
                        save_frames(frames, output_folder)
                        count_sample += 1
                    frames = []
                    count_frame = 0
                    timer = 0  # Reiniciar temporizador.
                    recording = False
                    cv2.putText(image, 'Muestra guardada', FONT_POS, FONT, FONT_SIZE, (0, 255, 0), 2)
                else:
                    recording = True
                    frames = []
                    count_frame = 0
                    timer = 0  # Iniciar temporizador.

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    word_name = "Buenos"
    word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
    capture_samples(word_path)
