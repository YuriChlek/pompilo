FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копіюємо код
COPY ./_OF-scalping-bot .

# Встановлюємо залежності
RUN pip install --upgrade pip && pip install -r requirements.txt

# Патчимо reconnecting_websocket.py після встановлення бібліотеки python-binance
RUN cp /app/patches/reconnecting_websocket.py /usr/local/lib/python3.12/site-packages/binance/ws/reconnecting_websocket.py
RUN cp /app/patches/streams.py /usr/local/lib/python3.12/site-packages/binance/ws/streams.py

# Запускаємо бота
CMD ["python", "main.py"]
