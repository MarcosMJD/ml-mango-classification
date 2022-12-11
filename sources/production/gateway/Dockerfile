FROM python:3.9.10-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock","./"]

RUN pipenv install --system --deploy

COPY ["gateway.py", "proto.py","./"]

# Flask
# ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:8080", "gateway:app" ]
ENTRYPOINT [ "gunicorn", "gateway:app", "--workers=4", "--worker-class=uvicorn.workers.UvicornWorker",   "--bind=0.0.0.0:8080"]

EXPOSE 8080