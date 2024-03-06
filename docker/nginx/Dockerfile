FROM nginx:1.17

# Install OpenSSL
RUN apt-get update && \
    apt-get install -y openssl

# Create a directory to store the certificates
RUN mkdir /certs

# Generate the certificates
RUN openssl genrsa 2048 > /certs/host.key \
    && openssl req -new -key /certs/host.key -subj "/C=JP/CN=localhost" > /certs/host.csr \
    && openssl x509 -days 3650 -req -signkey /certs/host.key < /certs/host.csr > /certs/host.crt

# Copy the Nginx configuration file
COPY ./docker/nginx/default.conf /etc/nginx/conf.d/default.conf

# Set the permissions for the certificates
RUN chmod 644 /certs/host.*

CMD ["nginx", "-g", "daemon off;"]