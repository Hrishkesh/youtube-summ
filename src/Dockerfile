FROM python:3.9.7

# Set the working directory in the container
WORKDIR usr/src/app

COPY ./requirements.txt /usr/src/app/requirements.txt

RUN pip install -r /usr/src/app/requirements.txt \
    && rm -rf /root/.cache/pip

COPY . /usr/src/

RUN ls -la

EXPOSE 8000