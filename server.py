# -*- coding: utf-8 -*-
import time
import logging
import sys
import argparse
import signal
from concurrent import futures

import grpc

import faissimageindex_pb2_grpc as pb2_grpc
from faiss_image_index import FaissImageIndex

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

def serve(args):
    logging.info('server loading')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    faiss_image_index = FaissImageIndex(args)
    pb2_grpc.add_ImageIndexServicer_to_server(faiss_image_index, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info('server started')

    # for docker heath check
    with open('/tmp/status', 'w') as f:
        f.write('started')

    def stop_serve(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop_serve)
    signal.signal(signal.SIGTERM, stop_serve)

    faiss_image_index.Migrate(None, None)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        faiss_image_index.save()
        logging.info('server stopped')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faiss Image Server')
    parser.add_argument('--log', help='log filepath')
    parser.add_argument('--rails_env', help='rails env')
    parser.add_argument('--train_count', type=int, default=100000, help='rails env')
    args = parser.parse_args()

    if args.log:
        handler = logging.FileHandler(filename=args.log)
    else:
        handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s - %(message)s')
    handler.setFormatter(formatter)
    root = logging.getLogger()
    level = args.rails_env == 'production' and logging.INFO or logging.DEBUG
    root.setLevel(level)
    root.addHandler(handler)

    serve(args)
