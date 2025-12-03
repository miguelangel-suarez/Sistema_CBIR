"""
--------------- EXTRACTORES.PY ----------------------

Este archivo Python es el responsable de coger todas las imágenes de entrenamiento (guardadas en la carpeta de "images"),
y crear los embeddings correspondientes a cada imagen según el extractor de características definido.
Además, este archivo esta encargado de crear las 5 tablas FAISS donde guardar el embedding de cada imagen según el
extractor de características pasado. Consecuentemente, los índices de dichas imagenes guardadas en FAISS se
almacenan dentro de un archivo CSV donde para cada índice guardar también el path de la imagen real (dentro de
los directorios de este proyecto) y la etiqueta final correspondiente a la clase a la que pertenece dicha imagen.

IMPORTANTE: Este archivo NO hay que ejecutarlo.

Sólo sirve para crear las tablas FAISS, el CSV y las funciones Python que utilizará el archivo Python principal "app.py".
Al ejecutar el archivo "app.py", se usarán las componentes creadas anteriormente, y se realizan llamadas a las funciones
de los extractores implementados.
"""

from PIL import Image
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
import torch
from torchvision import models, transforms as T
from torchvision.models import VGG19_Weights, Inception_V3_Weights
import cv2
import os

# Leemos Carpetas
IMAGES_FOLDER = Path("images")  # imágenes de entrenamiento
DB_PATH = Path("database")
DB_PATH.mkdir(exist_ok=True)

# Configuración general
DB_FILE = "db.csv"

# -------------------------
# EXTRACTORES
# -------------------------

# 1. Histograma RGB
def extract_rgb_histogram(img, size=(224,224)):
    img = img.resize(size)
    img_np = np.array(img)
    hist_r = np.histogram(img_np[:,:,0], bins=16, range=(0,255))[0]
    hist_g = np.histogram(img_np[:,:,1], bins=16, range=(0,255))[0]
    hist_b = np.histogram(img_np[:,:,2], bins=16, range=(0,255))[0]
    hist = np.concatenate([hist_r,hist_g,hist_b]).astype(np.float32)
    hist /= np.linalg.norm(hist) + 1e-6
    return hist.reshape(1,-1)


# 2. SIFT Descriptor
def extract_sift(img, size=(224,224)):
    # Convertir a escala de grises
    img = img.resize(size)
    img_gray = np.array(img.convert("L"))

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)

    if descriptors is None or len(descriptors) == 0:
        # Si no detecta keypoints, devolvemos un vector nulo
        descriptors = np.zeros((1, 128), dtype=np.float32)

    # Hacemos la media de todos los descriptores para obtener un vector fijo
    feat = np.mean(descriptors, axis=0)

    # Normalizamos
    feat = feat.astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-6)

    return feat.reshape(1, -1)


# 3. Segmentación (DeepLabV3)
deeplab_model = models.segmentation.deeplabv3_resnet50(weights="DEFAULT").eval()
seg_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
def extract_segmentation(img):
    img_t = seg_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = deeplab_model(img_t)['out']
        feat = torch.nn.functional.adaptive_avg_pool2d(output, (1,1)).squeeze().numpy()
    feat = feat.astype(np.float32)
    feat /= np.linalg.norm(feat) + 1e-6
    return feat.reshape(1,-1)


# 4. VGG19
vgg19_model = models.vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
vgg19_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
def extract_vgg19(img):
    img_t = vgg19_transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = vgg19_model(img_t).mean([2,3]).numpy()
    feat = feat.astype('float32')
    feat /= (np.linalg.norm(feat)+1e-6)
    return feat.reshape(1,-1)


# 5. InceptionV3
inception_model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
inception_model.eval()
inception_transform = T.Compose([
    T.Resize((299,299)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
def extract_inceptionv3(img):
    img_t = inception_transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = inception_model(img_t)
    feat = feat.detach().numpy().astype('float32')
    feat /= (np.linalg.norm(feat)+1e-6)
    return feat.reshape(1,-1)


# -------------------------
# FUNCIONES PARA GENERAR ÍNDICES FAISS
# -------------------------

def build_index(extractor_func, index_name):
    features = []
    image_files = []
    etiquetas = []
    
    for img_path in sorted(IMAGES_FOLDER.glob("*.*")):
        img = Image.open(img_path).convert("RGB")
        feat = extractor_func(img)
        features.append(feat)
        image_files.append(img_path.name)

        # Guardar etiquetas de las imágenes para el CSV
        name = os.path.splitext(os.path.basename(img_path))[0]
        etiqueta = name.rsplit('_', 1)[0]
        etiquetas.append(etiqueta)
    
    features = np.vstack(features).astype(np.float32)
    faiss.normalize_L2(features)
    
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    faiss.write_index(index, str(DB_PATH / index_name))

    # Guardar CSV
    dataframe = pd.DataFrame({'image': image_files, 'etiqueta': etiquetas})
    dataframe.to_csv(DB_PATH / DB_FILE, index=True)
    
    print(f"{index_name} creado con {len(image_files)} imágenes, shape: {features.shape}")


if __name__ == "__main__":
    build_index(extract_rgb_histogram, "extract_rgb_histogram.index")
    build_index(extract_segmentation, "extract_segmentation.index")
    build_index(extract_sift, "extract_sift.index")
    build_index(extract_vgg19, "extract_vgg19.index")
    build_index(extract_inceptionv3, "extract_inceptionv3.index")
