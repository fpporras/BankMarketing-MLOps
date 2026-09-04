<<<<<<< Updated upstream
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

=======
FROM python:3.14-slim

WORKDIR /app

>>>>>>> Stashed changes
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

<<<<<<< Updated upstream
COPY src ./src
COPY models ./models
=======
COPY . .
>>>>>>> Stashed changes

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]