FROM python:3.9.10-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock","./"]

RUN pipenv install --system --deploy

COPY ["gateway.py", "proto.py","./"]

# Flask
# ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:8080", "gateway:app" ]
# Gunicorn
# Use this with docker-compose
# ENTRYPOINT [ "gunicorn", "gateway:app", "--workers=4", "--worker-class=uvicorn.workers.UvicornWorker",   "--bind=0.0.0.0:8080"]
# Use this with Kubernetes
ENTRYPOINT [ "uvicorn", "gateway:app", "--host=0.0.0.0", "--port=8080"]

EXPOSE 8080