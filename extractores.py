from PIL import Image
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
import torch
from torchvision import models, transforms as T
from torchvision.models import VGG19_Weights, Inception_V3_Weights, MobileNet_V2_Weights
import os

# Carpetas
IMAGES_FOLDER = Path("images")  # imágenes de entrenamiento
DB_PATH = Path("database")
# DB_PATH.mkdir(exist_ok=True)

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


# 2. VGG19
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


# 3. InceptionV3
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
        # Solo usamos la salida principal, ignorando la auxiliar
        feat = inception_model(img_t)
    feat = feat.detach().numpy().astype('float32')
    feat /= (np.linalg.norm(feat)+1e-6)
    return feat.reshape(1,-1)


# 4. MobileNetV2
mobilenet_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features.eval()
mobilenet_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
def extract_mobilenet(img):
    img_t = mobilenet_transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = mobilenet_model(img_t).mean([2,3]).numpy()
    feat = feat.astype('float32')
    feat /= (np.linalg.norm(feat)+1e-6)
    return feat.reshape(1,-1)

# 5. Segmentación (DeepLabV3)
deeplab_model = models.segmentation.deeplabv3_resnet50(pretrained=True).eval()
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

# -------------------------
# FUNCIONES PARA GENERAR INDICES FAISS
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

        # Guardar etiquetas de las imágenes para meterlas en el CSV
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
    build_index(extract_vgg19, "extract_vgg19.index")
    build_index(extract_inceptionv3, "extract_inceptionv3.index")
    build_index(extract_mobilenet, "extract_mobilenet.index")
    build_index(extract_segmentation, "extract_segmentation.index")
