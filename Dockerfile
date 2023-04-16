FROM ubuntu:18.04
# FROM python:latest
WORKDIR /src 
COPY . /src 
CMD ["bash", "init_setup.sh"]



# FROM alpine:3.4

# RUN apk update && \
#        apk add curl && \
#        apk add vim && \
#        apk add git