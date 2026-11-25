FROM python:3.11-slim

# Чтобы вывод Python сразу попадал в логи контейнера и не создавались лишние .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Устанавливаем зависимости проекта
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения и модель
COPY . .

# Создаём папки под пользовательскую разметку заранее
RUN mkdir -p dataset_feedback/correct dataset_feedback/incorrect

# Порт, который будет слушать Flask
EXPOSE 5000

# Запускаем приложение (для продакшена позже можно заменить на gunicorn)
CMD ["python", "app.py"]