FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app/src

RUN python -m telco_churn.cli data \
    && python -m telco_churn.cli train \
    && python -m telco_churn.cli score

EXPOSE 8000
CMD ["uvicorn", "telco_churn.api:app", "--host", "0.0.0.0", "--port", "8000"]

