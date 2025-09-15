# Base layer (default dependencies) to use
FROM python:3.12
ENV LANG C.UTF-8

# Custom location to place code files
RUN mkdir -p /app/src
COPY serve_cpi_kserve.py /app/src/serve_cpi_kserve.py
COPY requirements.txt /app/src/requirements.txt
RUN pip3 install -r /app/src/requirements.txt

# Required to execute script
RUN chgrp -R nogroup /app && \
    chmod -R 770 /app

EXPOSE 9001
CMD ["uvicorn", "serve_cpi_kserve:app", "--host", "0.0.0.0", "--port", "9001"]
