import os
from capture_samples import capture_samples
from create_keypoints import create_keypoints
import prueba

root = os.getcwd()
words_path = os.path.join(root, "frame_actions")
data_path = os.path.join(root, "data")

# Pedir al usuario que ingrese la palabra
word_name = input("Ingrese la palabra a guardar en la cámara: ")

# Capturar muestras para la palabra ingresada
word_path = os.path.join(words_path, word_name)
capture_samples(word_path)

# Generar los keypoints de todas las palabras
for word_name in os.listdir(words_path):
    word_path = os.path.join(words_path, word_name)
    hdf_path = os.path.join(data_path, f"{word_name}.h5")
    print(f'Creando keypoints de "{word_name}"...')
    create_keypoints(word_path, hdf_path)

#root = os.getcwd()
#words_path = os.path.join(root, "action_frames")
#data_path = os.path.join(root, "data")

# CAPTURAR MUESTRAS PARA UNA PALABRA
# word_name = "censurado!"
# word_path = os.path.join(words_path, word_name)
# capture_samples(word_path)

# GENERAR LOS KEYPOINTS DE TODAS LAS PALABRAS
# for word_name in os.listdir(words_path):
#     word_path = os.path.join(words_path, word_name)
#     hdf_path = os.path.join(data_path, f"{word_name}.h5")
#     print(f'Creando keypoints de "{word_name}"...')
#     create_keypoints(word_path, hdf_path)