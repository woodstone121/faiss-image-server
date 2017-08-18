# -*- coding: utf-8 -*-
import time
import logging
import glob
import random
import shutil

import gevent
from gevent.pool import Pool
from gevent.threadpool import ThreadPool
import numpy as np
from tensorflow.python.lib.io import file_io

import faissimageindex_pb2 as pb2
import faissimageindex_pb2_grpc as pb2_grpc

from embeddings import ImageEmbeddingService
from faiss_index import FaissIndex, FaissShrinkedIndex

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def path_to_embedding(filepath):
    return np.fromstring(file_io.read_file_to_string(filepath, True),
            dtype=np.float32)


class FaissImageIndex(pb2_grpc.ImageIndexServicer):

    def __init__(self, args):
        self.save_filepath = args.save_filepath
        self.max_train_count = args.train_count
        t0 = time.time()
        self.embedding_service = ImageEmbeddingService(args.model)
        logging.info("embedding service loaded %.2f s" % (time.time() - t0))

        if file_io.file_exists(self.save_filepath):
            self.faiss_index = self._new_index()
            t0 = time.time()
            self.faiss_index.restore(self.save_filepath)
            logging.info("%d items restored %.2f s", self.faiss_index.ntotal(), time.time() - t0)
        else:
            self.faiss_index = self._new_trained_index()

    def Migrate(self, request, context):
        logging.info('Migrating...')
        path = 'embeddings'
        count = 0
        pos = len(path) + 1

        def move_file(filepath):
            id = int(filepath[pos:-4])
            to_path = self._get_filepath(id, mkdir=True)
            return shutil.move(filepath, to_path)

        t0 = time.time()
        start_t0 = t0

        pool = ThreadPool(12)
        for filepath in glob.iglob('%s/*.emb' % path):
            pool.spawn(move_file, filepath)
            count += 1
            if count % 10000 == 0:
                gevent.wait()
                logging.info("migrating %d, %.2f s", count, time.time() - t0)
                t0 = time.time()
        gevent.wait()

        logging.info("Migrated %d emb files to sub dir %.2f s", count, time.time() - start_t0)
        return pb2.SimpleReponse(message='Migrated %d' % count)

    def _new_index(self, nlist=100):
        d = self.embedding_service.dim()
        faiss_index = FaissShrinkedIndex(d, nlist=nlist)
        logging.info(faiss_index.__class__)
        logging.info("nlist: %d", nlist)
        return faiss_index

    def Train(self, request, context):
        pre_index = self.faiss_index
        self.faiss_index = self._new_trained_index()
        pre_index.reset()
        return pb2.SimpleReponse(message='Trained')

    def _path_to_xb(self, paths):
        d = self.embedding_service.dim()
        xb = np.ndarray(shape=(len(paths), d), dtype=np.float32)
        p = Pool(12)
        for i, emb in enumerate(p.imap(path_to_embedding, paths)):
            xb[i] = emb
        return xb

    def _new_trained_index(self):
        def path_to_id(filepath):
            pos = filepath.rindex('/') + 1
            return int(filepath[pos:-4])

        logging.info("File loading...")
        t0 = time.time()
        all_filepaths = list(glob.iglob('embeddings/*/*.emb'))
        total_count = len(all_filepaths)
        logging.info("%d files %.3f s", total_count, time.time() - t0)

        train_count = min(total_count, self.max_train_count)
        if train_count <= 0:
            return self._new_index()

        random.shuffle(all_filepaths)

        filepaths = all_filepaths[:train_count]
        t0 = time.time()
        xb = self._path_to_xb(filepaths)
        logging.info("%d embeddings loaded %.3f s", xb.shape[0], time.time() - t0)

        ids = np.array(list(map(path_to_id, filepaths)), dtype=np.int64)

        if train_count < 40000:
            d = self.embedding_service.dim()
            faiss_index = FaissIndex(d)
            faiss_index.add(xb, ids)
            return faiss_index

        faiss_index = self._new_index(nlist=int(train_count / 800))

        logging.info("Training...")
        t0 = time.time()
        faiss_index.train(xb, ids)
        logging.info("trained %.3f s", time.time() - t0)

        t0 = time.time()
        faiss_index.add(xb, ids)
        logging.info("added %.3f s", time.time() - t0)

        if total_count > train_count:
            for filepaths in chunks(all_filepaths[train_count:], 20000):
                t0 = time.time()
                xb = self._path_to_xb(filepaths)
                ids = np.array(map(path_to_id, filepaths), dtype=np.int64)
                faiss_index.add(xb, ids)
                logging.info("%d embeddings added %.3f s", xb.shape[0], time.time() - t0)
            logging.info("Total %d embeddings added", faiss_index.ntotal())
        return faiss_index

    def Save(self, request, context):
        self.save()
        return pb2.SimpleReponse(message='Saved')

    def save(self):
        t0 = time.time()
        self.faiss_index.save(self.save_filepath)
        logging.info("index saved to %s, %.3f s", self.save_filepath, time.time() - t0)

    def Add(self, request, context):
        if self._more_recent_emb_file_exists(request):
            return pb2.SimpleReponse(message='Already added, %s!' % request.id)

        embedding = self.fetch_embedding(request)
        if embedding is None:
            return pb2.SimpleReponse(message='No embedding, id: %d, url: %s' % (request.id, request.url))

        embedding = np.expand_dims(embedding, 0)
        ids = np.array([request.id], dtype=np.int64)
        self.faiss_index.replace(embedding, ids)

        return pb2.SimpleReponse(message='Added, %s!' % request.id)

    def Import(self, request, context):
        def get_mtime(filepath):
            if file_io.file_exists(filepath):
                return file_io.stat(filepath).mtime_nsec
            return None

        def is_new_emb(id, filepath):
            origin_mtime = get_mtime(self._get_filepath(id))
            if origin_mtime is None:
                return True
            new_mtime = get_mtime(filepath)
            return origin_mtime < new_mtime

        logging.info("Importing..")
        all_filepaths = list(glob.iglob('%s/*.emb' % request.path))

        total_count = len(all_filepaths)
        if total_count <= 0:
            logging.info("No files for importing!")
            return pb2.SimpleReponse(message='No files for importing!')

        logging.info("Importing files count: %d" % total_count)

        pos = len(request.path) + 1
        def path_to_id(filepath):
            return int(filepath[pos:-4])

        for filepaths in chunks(all_filepaths, 10000):
            t0 = time.time()

            ids = map(path_to_id, filepaths)
            ids_filepaths = [(id, filepath) for id, filepath in zip(ids, filepaths) if is_new_emb(id, filepath)]

            xb = self._path_to_xb([filepath for _, filepath in ids_filepaths])
            ids = np.array([id for id, _ in ids_filepaths], dtype=np.int64)
            self.faiss_index.replace(xb, ids)

            for id, filepath in ids_filepaths:
                file_io.rename(filepath, self._get_filepath(id, mkdir=True), overwrite=True)

            logging.info("%d embeddings added %.3f s", xb.shape[0], time.time() - t0)
        return pb2.SimpleReponse(message='Imported, %d!' % total_count)

    def _more_recent_emb_file_exists(self, request):
        filepath = self._get_filepath(request.id)
        if not file_io.file_exists(filepath):
            return False
        file_ts = file_io.stat(filepath).mtime_nsec / 1000000000
        return file_ts >= request.created_at_ts

    def fetch_embedding(self, request):
        t0 = time.time()
        embedding = self.embedding_service.get_embedding(request.url)
        if embedding is None:
            return
        logging.debug("embedding fetched %d, %.3f s", request.id, time.time() - t0)
        filepath = self._get_filepath(request.id, mkdir=True)
        file_io.write_string_to_file(filepath, embedding.tostring())
        return embedding

    def Fetch(self, request, context):
        total_count = len(request.items)
        fetched_count = 0

        results = []

        pool = ThreadPool(12)
        for item in request.items:
            result = pool.spawn(self.fetch_embedding, item)
            results.append(result)
        gevent.wait()

        for result in results:
            if result.get() is not None:
                fetched_count += 1

        return pb2.SimpleReponse(message='Fetched, %d of %d!' % (fetched_count, total_count))

    def _get_filepath(self, id, mkdir=False):
        path = 'embeddings/%d' % int(id / 10000)
        if mkdir and not file_io.file_exists(path):
            file_io.create_dir(path)
        return '%s/%d.emb' % (path, id)

    def Search(self, request, context):
        filepath = self._get_filepath(request.id) 
        embedding = path_to_embedding(filepath)
        embedding = np.expand_dims(embedding, 0)
        D, I = self.faiss_index.search(embedding, request.count)
        return pb2.SearchReponse(ids=I[0], scores=D[0])

    def Info(self, request, context):
        return pb2.SimpleReponse(message='%s' % self.faiss_index.ntotal())

    def Remove(self, request, context):
        ids = np.array([request.id], dtype=np.int64)
        self.faiss_index.remove_ids(ids)

        filepath = self._get_filepath(request.id) 
        if file_io.file_exists(filepath):
            file_io.delete_file(filepath)
            return pb2.SimpleReponse(message='Removed, %s!' % request.id)

        return pb2.SimpleReponse(message='Not existed, %s!' % request.id)

