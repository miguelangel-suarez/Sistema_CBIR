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

# Importar extractores
from extractores import (
    extract_rgb_histogram,
    extract_vgg19,
    extract_inceptionv3,
    extract_mobilenet,
    extract_segmentation
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(layout="wide")

device = torch.device('cpu')

FILES_PATH = str(pathlib.Path().resolve())

# Path in which the images should be located
IMAGES_PATH = os.path.join(FILES_PATH, 'images')
# Path in which the database should be located
DB_PATH = os.path.join(FILES_PATH, 'database')

DB_FILE = 'db.csv'  # CSV con nombres de imagenes


def get_image_list():
    df = pd.read_csv(os.path.join(DB_PATH, DB_FILE))
    image_list = list(df.image.values)
    return image_list

def retrieve_image(img_query, feature_extractor, n_imgs=11):
    if feature_extractor == 'Extractor 1':
        model_feature_extractor = extract_rgb_histogram
        indexer = faiss.read_index(os.path.join(DB_PATH, 'extract_rgb_histogram.index'))
    elif feature_extractor == 'Extractor 2':
        model_feature_extractor = extract_vgg19
        indexer = faiss.read_index(os.path.join(DB_PATH, 'extract_vgg19.index'))
    elif feature_extractor == 'Extractor 3':
        model_feature_extractor = extract_inceptionv3
        indexer = faiss.read_index(os.path.join(DB_PATH, 'extract_inceptionv3.index'))
    elif feature_extractor == 'Extractor 4':
        model_feature_extractor = extract_mobilenet
        indexer = faiss.read_index(os.path.join(DB_PATH, 'extract_mobilenet.index'))
    elif feature_extractor == 'Extractor 5':
        model_feature_extractor = extract_segmentation
        indexer = faiss.read_index(os.path.join(DB_PATH, 'extract_segmentation.index'))
    else:
        raise ValueError(f"Extractor '{feature_extractor}' no definido")

    # Extraer características y normalizar
    embeddings = model_feature_extractor(img_query)
    vector = np.float32(embeddings)
    faiss.normalize_L2(vector)

    # Buscar en el índice
    distancias, indices = indexer.search(vector, k=n_imgs)
    return distancias[0], indices[0]

def main():
    st.title('CBIR IMAGE SEARCH')
    
    col1, col2 = st.columns(2)

    with col1:
        st.header('QUERY')

        st.subheader('Choose feature extractor')
        option = st.selectbox('.', (
            'Extractor 1',  # RGB Histogram
            'Extractor 2',  # VGG19
            'Extractor 3',  # InceptionV3
            'Extractor 4',  # MobileNet
            'Extractor 5'   # Segmentation
        ))

        st.subheader('Upload image')
        img_file = st.file_uploader(label='.', type=['png', 'jpg'])

        if img_file:
            img = Image.open(img_file)
            # Obtener imagen recortada
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')
            
            st.write("Preview")
            _ = cropped_img.thumbnail((150,150))
            st.image(cropped_img)

    with col2:
        st.header('RESULT')
        if img_file:
            st.markdown('**Retrieving .......**')
            start = time.time()

            dist, retriev = retrieve_image(cropped_img, option, n_imgs=11)
            image_list = get_image_list()

            end = time.time()
            st.markdown('**Finish in ' + str(end - start) + ' seconds**')

            col3, col4 = st.columns(2)

            with col3:
                image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[0]]))
                st.image(image, use_container_width = 'always')
                st.write(f"Distancia: {round(dist[0], 3):.3f}, con Indice: {retriev[0]}")

            with col4:
                image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[1]]))
                st.image(image, use_container_width = 'always')
                st.write(f"Distancia: {round(dist[1], 3):.3f}, con Indice: {retriev[1]}")

            col5, col6, col7 = st.columns(3)

            with col5:
                for u in range(2, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_container_width = 'always')
                    st.write(f"Distancia: {round(dist[u], 3):.3f}, con Indice: {retriev[u]}")

            with col6:
                for u in range(3, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_container_width = 'always')
                    st.write(f"Distancia: {round(dist[u], 3):.3f}, con Indice: {retriev[u]}")

            with col7:
                for u in range(4, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_container_width = 'always')
                    st.write(f"Distancia: {round(dist[u], 3):.3f}, con Indice: {retriev[u]}")


if __name__ == '__main__':
    main()
