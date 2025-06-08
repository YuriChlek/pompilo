FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копіюємо код
COPY ./_OF-rain-bot .

# Встановлюємо залежності
RUN pip install --upgrade pip && pip install -r requirements.txt

# Запускаємо бота
CMD ["python", "main.py"]
