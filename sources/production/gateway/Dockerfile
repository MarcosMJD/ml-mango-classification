FROM python:3.9.10-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock","./"]

RUN pipenv install --system --deploy

COPY ["gateway.py", "proto.py","./"]

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9000", "gateway:app" ]

EXPOSE 9000