"""
--------------------- APP.PY -----------------

Este archivo es el responsable de generar una interfaz de usuario mediante la librería de "streamlit" para así dar
funcionalidad al proyecto completo.

IMPORTANTE: Ejecutar este archivo:
          1. Primero, situarse en el directorio principal de "Proyecto_CBIR".
          2. Instalar todas las dependencias necesarias, presentes en el archivo "requirements.txt".
          3. Descargar las 2 carpetas de imágenes "test" y "images".
          4. Ejecutar el siguiente comando a través de la terminal:
                    "streamlit run app.py"
          5. Cuando se ejecute la interfaz de STREAMLIT, seleccionar el extractor a usar, insertar una de las imágenes
             disponibles en el directorio de "test" y comprobar el funcionamiento del programa.


"App.py" hace uso de los archivos contenidos en la carpeta de "database", y de las funciones Python alojadas en el
archivo de "extractores.py" para realizar la creación de los embeddings correspondientes al extractor seleccionado.

Las imágenes que se usan para realizar las consultas desde la interfaz son las imágenes de la carpeta de "test", que se
puede encontrar dentro del enlace de Google Drive (Junto con el resto de imágenes usadas para el entrenamiento, las
cuáles hay que descargarse también para que la interfaz pueda mostrar las imágemes más parecidas a la imagen
 de consulta):

          - https://drive.google.com/drive/folders/1p3dLfA0RKP7PGz6-KgcPmk8MWoNitVAy?usp=drive_link
"""

import torch
import faiss
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import os
import time
import streamlit as st
from streamlit_cropper import st_cropper

# Importar extractores actualizados
from extractores import (
    extract_rgb_histogram,
    extract_vgg19,
    extract_inceptionv3,
    extract_sift,
    extract_segmentation
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(layout="wide")

device = torch.device('cpu')

FILES_PATH = str(pathlib.Path().resolve())

# Paths
IMAGES_PATH = os.path.join(FILES_PATH, 'images')
DB_PATH = os.path.join(FILES_PATH, 'database')
DB_FILE = 'db.csv'  # CSV con nombres de imágenes y etiquetas


# ===================== FUNCIONES AUXILIARES =====================

def get_image_list():
    df = pd.read_csv(os.path.join(DB_PATH, DB_FILE))
    return list(df.image.values)


def get_label_list(retriev: list):
    df = pd.read_csv(os.path.join(DB_PATH, DB_FILE))
    all_labels = list(df.etiqueta.values)
    return [all_labels[i] for i in retriev]


def get_percentage(label_list: list, img_name: str):
    df = pd.read_csv(os.path.join(DB_PATH, DB_FILE))
    row = df[df['image'] == img_name]
    real_label = row['etiqueta'].values[0] if not row.empty else img_name.rsplit('_', 1)[0]
    counter = label_list.count(real_label)
    percentage = (counter / len(label_list)) * 100
    return round(percentage, 2)


# ===================== FUNCIÓN PRINCIPAL DE RETRIEVAL =====================

def retrieve_image(img_query, feature_extractor, n_imgs=11):
    if feature_extractor == 'RGB Histogram':
        model_feature_extractor = extract_rgb_histogram
        indexer = faiss.read_index(os.path.join(DB_PATH, 'extract_rgb_histogram.index'))
    elif feature_extractor == 'SIFT':
        model_feature_extractor = extract_sift
        indexer = faiss.read_index(os.path.join(DB_PATH, 'extract_sift.index'))
    elif feature_extractor == 'DeepLabV3':
        model_feature_extractor = extract_segmentation
        indexer = faiss.read_index(os.path.join(DB_PATH, 'extract_segmentation.index'))
    elif feature_extractor == 'VGG19':
        model_feature_extractor = extract_vgg19
        indexer = faiss.read_index(os.path.join(DB_PATH, 'extract_vgg19.index'))
    elif feature_extractor == 'InceptionV3':
        model_feature_extractor = extract_inceptionv3
        indexer = faiss.read_index(os.path.join(DB_PATH, 'extract_inceptionv3.index'))
    else:
        raise ValueError(f"Extractor '{feature_extractor}' no definido")

    embeddings = model_feature_extractor(img_query)
    vector = np.float32(embeddings)
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    faiss.normalize_L2(vector)
    distances, indices = indexer.search(vector, k=n_imgs)
    return distances[0], indices[0]


# ===================== INTERFAZ STREAMLIT =====================

def main():
    st.title('CBIR IMAGE SEARCH')

    col1, col2 = st.columns(2)

    # Columna izquierda: consulta
    with col1:
        st.header('QUERY')

        st.subheader('Choose feature extractor')
        option = st.selectbox(' ', (
            'RGB Histogram',
            'SIFT',
            'DeepLabV3',
            'VGG19',
            'InceptionV3'
        ))

        st.subheader('Upload image')
        img_file = st.file_uploader(label=' ', type=['png', 'jpg', 'jpeg'])

        if img_file:
            img = Image.open(img_file)
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')

            st.write("Preview")
            _ = cropped_img.thumbnail((150, 150))
            st.image(cropped_img)

    # Columna derecha: resultados
    with col2:
        st.header('RESULTS')

        if img_file:
            st.markdown('**Retrieving .......**')
            start = time.time()

            dist, retriev = retrieve_image(cropped_img, option, n_imgs=11)
            image_list = get_image_list()
            label_list = get_label_list(retriev)
            porcentaje = get_percentage(label_list, img_file.name)

            end = time.time()
            st.markdown(f'**Finish in {round(end - start, 3)} seconds**')
            st.subheader(f"Accuracy: {porcentaje}%")

            # Mostrar las 11 imágenes recuperadas con etiqueta y distancia
            col3, col4 = st.columns(2)
            with col3:
                st.write(f"{label_list[0]} - Distancia: {dist[0]:.3f}")
                image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[0]]))
                st.image(image, use_container_width=True)

            with col4:
                st.write(f"{label_list[1]} - Distancia: {dist[1]:.3f}")
                image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[1]]))
                st.image(image, use_container_width=True)

            col5, col6, col7 = st.columns(3)
            with col5:
                for u in range(2, 11, 3):
                    st.write(f"{label_list[u]} - Distancia: {dist[u]:.3f}")
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_container_width=True)
            with col6:
                for u in range(3, 11, 3):
                    st.write(f"{label_list[u]} - Distancia: {dist[u]:.3f}")
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_container_width=True)
            with col7:
                for u in range(4, 11, 3):
                    st.write(f"{label_list[u]} - Distancia: {dist[u]:.3f}")
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_container_width=True)


if __name__ == '__main__':
    main()
