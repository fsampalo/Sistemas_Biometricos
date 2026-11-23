# Sistema de reconocimiento facial EBGM — imagen lista para usar.
#
#   docker build -t ebgm .
#   docker run --rm ebgm                                  # demo de los 9 métodos
#   docker run --rm ebgm python scripts/evaluar.py                # evaluación (fase 2)
#   docker run --rm ebgm python scripts/evaluar_ataques.py        # ataques (fase 2)
#   docker run --rm -v "$PWD/fotos:/fotos" ebgm python -m reconocimiento.sistema identificar --foto /fotos/x.jpg
#
# El build entrena el modelo EBGM e inscribe las 50 personas, así que el
# contenedor arranca con la BD lista. Para la clave HMAC de la BD en producción,
# pásala por entorno:  docker run -e EBGM_CLAVE_BD=<hex de 64 caracteres> ...
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN if [ ! -d datos/Dataset ] && [ -f datos/Dataset.tar.gz ]; then tar -xzf datos/Dataset.tar.gz -C datos; fi \
 && python scripts/download_models.py \
 && python scripts/inscribir_dataset.py

CMD ["python", "scripts/demo.py"]
