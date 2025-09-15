# Specify which base layers (default dependencies) to use
FROM python:3.12

# Creates directory within your Docker image
RUN mkdir -p /app/src/
# Don't place anything in below folders yet, just create them
RUN mkdir -p /app/data/
RUN mkdir -p /app/model/

# Copies files from your local system TO the Docker image
COPY requirements.txt /app/src/
COPY train_cpi_pipeline.py /app/src/

# Installs dependencies within your Docker image
RUN pip3 install -r /app/src/requirements.txt

# Enable permission to execute anything inside the folder app
RUN chgrp -R 65534 /app && \
    chmod -R 777 /app

# Default command (AI Core can override with args)
CMD ["python", "/app/src/train_cpi_pipeline.py"]
