# CBIR

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

Para el correcto funcionamiento del programa, se DEBERÁ descargar las 2 carpetas de "images" y "test", y situarlas al
mismo nivel que el resto de carpetas y archivos.


## Ejecución

Para ejecutar la aplicación, sólo se debe ejecutar el archivo de "app.py".
Para ello, use el siguiente comando:

```bash
streamlit run app.py
```

La aplicación se ejecutará en <http://localhost:8501>