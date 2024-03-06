FROM mysql:5.7

COPY ./docker/db/mysql5_7/my.cnf /etc/mysql/conf.d/my.cnf

RUN mkdir /var/log/mysql \
    && chown mysql:mysql /var/log/mysql \
    && chown mysql:mysql /var/lib/mysql \
    && chmod 777 /var/lib/mysql
RUN groupmod -g 1000 mysql && usermod -u 1000 -g 1000 mysql
