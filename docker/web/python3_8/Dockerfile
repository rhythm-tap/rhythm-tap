FROM python:3.8

# 自己証明書
RUN apt-get update && \
    apt-get install -y openssl && \
    openssl genrsa 2048 > host.key && \
    openssl req -new -key host.key -subj "/C=JP/CN=localhost" > host.csr && \
    openssl x509 -days 3650 -req -signkey host.key < host.csr > host.crt && \
    mkdir /certs && \
    mv host.key host.crt /certs/

# タイムゾーンの設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

ADD ./src/requirements.txt /app/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ADD ./src /app/

CMD ["python", "app.py"]
