FROM python:3.12-slim

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY README.md .

ENTRYPOINT ["python"]