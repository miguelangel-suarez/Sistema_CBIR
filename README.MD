# CBIR

## Requisitos

- Python >= 3.8

## Instalación

1. Clona este repositorio en tu máquina local:

    ```bash
    git clone https://github.com/miguelangel-suarez/Proyecto_CBIR
    cd cbir
    ```


2. Instala las dependencias dentro de un entorno virtual:

    ```bash
    pip install -r requirements.txt
    ```
3. Descargar desde Google Drive las carpetas correspondientes a las imágenes
de entrenamiento (carpeta "images") y a las imágenes de test (carpeta "test").
- https://drive.google.com/drive/folders/1p3dLfA0RKP7PGz6-KgcPmk8MWoNitVAy?usp=drive_link

Añadir las 2 carpetas de imágenes al mismo nivel que el resto de los archivos de este repositorio.


## Ejecución

Para ejecutar la aplicación, sólo se debe ejecutar el archivo de "app.py".
Para ello, use el siguiente comando:

```bash
streamlit run app.py
```

La aplicación se ejecutará en <http://localhost:8501>