FROM python:3.11-slim

WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn

# Копируем остальные файлы
COPY pharm-nlp-service.py .

# Запускаем приложение
CMD ["uvicorn", "pharm-nlp-service:app", "--host", "0.0.0.0", "--port", "8000"]