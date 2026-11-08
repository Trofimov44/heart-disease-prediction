# Базовый образ
FROM python:3.13-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Копируем зависимости и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и модель
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Порт FastAPI
EXPOSE 8000

# Запуск FastAPI
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]