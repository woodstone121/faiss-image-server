FROM daangn/faiss:py3

ENV GRPC_PYTHON_VERSION 1.4.0
RUN python3 -m pip install --upgrade pip
RUN pip3 install grpcio==${GRPC_PYTHON_VERSION} grpcio-tools==${GRPC_PYTHON_VERSION}

RUN pip3 install tensorflow==1.2.0
RUN pip3 install pillow==4.2.1

RUN mkdir -p /app
WORKDIR /app

#ONBUILD COPY requirements.txt /usr/src/app/
#ONBUILD RUN pip install --no-cache-dir -r requirements.txt

RUN pip3 install gevent==1.2.2

# https://tensorflow.blog/2017/05/12/tf-%EC%84%B1%EB%8A%A5-%ED%8C%81-winograd-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%84%A4%EC%A0%95/
ENV TF_ENABLE_WINOGRAD_NONFUSED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

ENTRYPOINT ["python3"]
CMD ["server.py"]

HEALTHCHECK --interval=3s --timeout=2s \
  CMD ls /tmp/status || exit 1

RUN mkdir nets && cd nets && \
      wget https://github.com/tensorflow/models/raw/master/slim/nets/__init__.py && \
      wget https://github.com/tensorflow/models/raw/master/slim/nets/inception_utils.py && \
      wget https://github.com/tensorflow/models/raw/master/slim/nets/inception_v4.py

COPY *.py /app/
