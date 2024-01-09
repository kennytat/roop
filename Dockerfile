FROM python:3.9-bullseye

RUN apt update -y && apt upgrade -y

RUN apt -y install -qq aria2 wget curl git libgl1 ffmpeg

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN rm -rf /root/.cache/pip && rm -rf /var/cache/apt/*

COPY . .

EXPOSE 7890

CMD python /app/webui.py