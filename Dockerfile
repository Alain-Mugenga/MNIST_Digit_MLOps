FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY static ./static
COPY data ./data
RUN mkdir -p /opt/initial_storage && STORAGE_DIR=/opt/initial_storage python -m src.model
COPY docker_start.sh ./docker_start.sh
RUN chmod +x docker_start.sh
ENV STORAGE_DIR=/app/storage
EXPOSE 8000
CMD ["./docker_start.sh"]
