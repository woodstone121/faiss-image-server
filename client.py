from __future__ import print_function
import os

import grpc

import faissimageindex_pb2 as pb2
import faissimageindex_pb2_grpc as pb2_grpc


def run():
  host = 'localhost'
  channel = grpc.insecure_channel('localhost:50051')
  stub = pb2_grpc.ImageIndexStub(channel)

  response = stub.Info(pb2.Empty())
  print("client received: " + response.message)


if __name__ == '__main__':
  run()
