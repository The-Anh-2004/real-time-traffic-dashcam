FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libgl1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy opencv-python-headless playsound==1.2.2

COPY . .

CMD ["python", "run.py"]