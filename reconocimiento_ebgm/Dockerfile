FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# La BD inscrita (datos/bd.csv) y el FBG (modelos/fbg.npz) se incluyen en la imagen.
# Por defecto se identifica la foto que se monte en /app/entrada.jpg
ENTRYPOINT ["python", "cli.py"]
CMD ["listar"]
