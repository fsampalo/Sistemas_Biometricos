# Sistema de reconocimiento facial EBGM — imagen lista para usar.
#
#   docker build -t ebgm .
#   docker run --rm ebgm                      # demo de los 9 métodos
#   docker run --rm ebgm python evaluar.py    # métricas
#   docker run --rm ebgm python evaluar_ataques.py   # análisis de seguridad
#
# El Dataset de famosos va incluido en la imagen (se copia en el build). Para
# usar otro dataset, móntalo:  docker run --rm -v "$PWD/Dataset:/app/Dataset" ebgm
FROM python:3.11-slim

# OpenCV headless necesita estas librerías del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias primero (mejor cacheo de capas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código y datos
COPY . .

# Descargar el modelo del detector (YuNet) en el build
RUN python download_models.py

# Inscribir famosos + cigala y calibrar el umbral durante el build, para que el
# contenedor arranque con la BD lista.
RUN python inscribir_dataset.py && python evaluar.py --guardar-umbral || true

CMD ["python", "demo.py"]
