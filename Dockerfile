FROM python:3.10-slim

# Evitar archivos pyc y buffer en logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema para procesar imágenes (OpenCV, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Copiar e instalar requerimientos primero (para aprovechar caché de Docker)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 2. Copiar el código fuente
COPY app.py .
COPY extractores.py .

# 3. Copiar la base de datos vectorial
# Esto mete tus tablas FAISS dentro de la imagen final
COPY database/ ./database/

# Exponer puerto de Streamlit
EXPOSE 8501

# Chequeo de salud
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando de ejecución
ENTRYPOINT ["streamlit", "run", "app.py"]