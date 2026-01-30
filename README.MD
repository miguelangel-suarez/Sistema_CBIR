# SISTEMA DE RECUPERACIÓN DE IMÁGENES BASADO EN CONTENIDO

El objetivo de este proyecto es desarrollar un sistema de Recuperación de Imágenes Basado en Contenido (CBIR) para buscar imágenes de una base de datos vectorial basandose simplemente en los patrones y estructuras presentes en las imágenes, donde no se depende de etiquetas o metadatos en las imágenes. Buscando similitudes en el contenido de distintas imágenes, el objetivo es encontrar las imágenes más parecidas a una dada. 
Para este proyecto se utilizó el dataset "Birds 20 Species" de Kaggle, de las cuáles se utilizaron 5 clases de aves para desarrollar el sistema.

Para la creación de los embeddings de cada imagen que se guardaran en una base de datos vectorial, se diseñaron 5 extractores de características distintos: histograma RGB, extractor SIFT, InceptionV3 y VGG19. 

De cada imagen se obtiene un embedding distinto (según el extractor de características seleccionado), y gracias a la librería de FAISS para Python, se guarda dicho embedding en una tabla FAISS. Posteriormente, con una imagen nueva, se selecciona un extractor y obtenido su embedding se realiza una consulta a su tabla correspondiente, y se devuelven las imágenes con el embedding más similar.

Como interfaz de utiliza la librería de Streamlit de Python, para poder interactuar con el sistema y así realizar las consultas con las imágenes nuevas.

Este proyecto tiene un alto énfasis experimental, donde el objetivo es comparar el rendimiento de cada extractor a la hora de recoger y detectar las formas y estructuras más importantes de cada imagen. Dicho rendimiento se observa directamente en la precisión de cada modelo donde los 2 modelos basados en redes neuronales como VGG19 e InceptionV3, especializados en reconocimiento de imágenes, son muy superiores en comparación con las otras técnicas evaluadas.


## Requisitos

- Python >= 3.8

## Instalación

1. Clona este repositorio en tu máquina local:

    ```bash
    git clone https://github.com/miguelangel-suarez/Proyecto_CBIR
    cd Proyecto_CBIR 
   ```

2. Crear un entorno virtual y activarlo (desde Windows en nuestro caso):

   ```bash
   python -m venv venv
   venv/Scripts/activate
   ```

3. Instala las dependencias dentro del entorno virtual:

    ```bash
    pip install -r requirements.txt
    ```
4. Descargar desde Google Drive las carpetas correspondientes a las imágenes
de entrenamiento (carpeta "images") y a las imágenes de test (carpeta "test").
- https://drive.google.com/drive/folders/1p3dLfA0RKP7PGz6-KgcPmk8MWoNitVAy?usp=drive_link

Este repositorio ya ofrece la carpeta "images", sin embargo para obtener las imágenes reservadas para evaluar dicho proyecto, se deberá acceder al enlace anterior y descargar la carpeta de "test". Dicha carpeta se deberá situar al mismo nivel que el resto de los archivos de este repositorio.


## Ejecución

Para ejecutar la aplicación, sólo se debe ejecutar el archivo de "app.py".
Para ello, use el siguiente comando:

```bash
streamlit run app.py
```

La aplicación se ejecutará en <http://localhost:8501>


## Docker

Para su despliegue en contenedores de Docker, se ofrecen los archivos Dockerfile y docker-compose.yml. Para su despliegue, se usa el siguiente comando:

```bash
docker compose up -d
```

Con ello, se creará la correspodiente imagen y contenedor donde se ejecuta dicho programa.
