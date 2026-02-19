# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание пользователя для безопасности
RUN useradd -m -u 1000 ai_user && chown -R ai_user:ai_user /app
USER ai_user

# Создание директорий для данных
RUN mkdir -p /app/data /app/logs /app/models /app/uploads

# Экспорт портов
EXPOSE 8000

# Запуск приложения
CMD ["python", "-m", "app.main"]