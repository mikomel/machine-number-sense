FROM pytorch/pytorch:1.7.0-cuda10.1-cudnn7-devel

RUN mkdir /app
RUN mkdir /app/logs
RUN mkdir /app/datasets

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY mns /app/mns

WORKDIR /app
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:."
ENTRYPOINT ["python", "mns/module.py"]
