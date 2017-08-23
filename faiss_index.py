import logging
import time

import faiss

class FaissIndex:
    def __init__(self, d):
        self.index = faiss.IndexFlatIP(d)
        self.index2 = faiss.IndexIDMap(self.index)

    def train(self, xb, ids):
        pass

    def replace(self, xb, ids):
        self.remove_ids(ids)
        return self.index2.add_with_ids(xb, ids)

    def add(self, xb, ids):
        return self.index2.add_with_ids(xb, ids)

    def search(self, xq, k=10):
        return self.index2.search(xq, k)

    def ntotal(self):
        return self.index.ntotal

    def remove_ids(self, ids):
        return self.index2.remove_ids(ids)

    def restore(self, filepath):
        self.index2 = faiss.read_index(filepath)

    def save(self, filepath):
        pass

    def reset(self):
        self.index.reset()
        self.index = None
        self.index2.reset()
        self.index2 = None


class FaissTrainIndex(FaissIndex):

    def train(self, xb, ids):
        assert not self.index2.is_trained
        self.index2.train(xb)
        assert self.index2.is_trained

    def ntotal(self):
        return self.index2.ntotal

    def save(self, filepath):
        faiss.write_index(self.index2, filepath)

    def reset(self):
        self.index2.reset()
        self.index2 = None


class FaissFastIndex(FaissTrainIndex):
    def __init__(self, d):
        nlist = 100
        quantizer = faiss.IndexFlatL2(d)
        self.index2 = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

        # https://github.com/facebookresearch/faiss/wiki/Faiss-code-structure#object-ownership
        self.index2.own_fields = True
        quantizer.this.disown()

        self.index2.nprobe = 10


class FaissShrinkedIndex(FaissTrainIndex):
    # nlist  numCentroids
    def __init__(self, d, nlist=100):
        m = 8 # number of subquantizers
        nlist = min(nlist, 4096)

        quantizer = faiss.IndexFlatL2(d)
        self.index2 = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

        #self.index2.own_fields = True
        #quantizer.this.disown()
        self.quantizer = quantizer

        self.index2.nprobe = 32

    def train(self, xb, ids):
        #self.index2.polysemous_ordering = True
        super(FaissShrinkedIndex, self).train(xb, ids)
        #self.index2.polysemous_ht = 54 # the Hamming threshold

    def reset(self):
        self.quantizer.reset()
        self.quantizer = None
        self.index2.reset()
        self.index2 = None


class FaissShrinkedIndex2(FaissTrainIndex):
    def __init__(self, d, nlist=100):
        self.index = faiss.IndexPQ(d, 16, 8)
        self.index.nprobe = 10
        self.index2 = faiss.IndexIDMap(self.index)


class FaissOPQIndex(FaissTrainIndex):
    def __init__(self, d, nlist=100):
        self.index2 = faiss.index_factory(d, 'OPQ32_128,IVF4096,PQ32')
        self.index2.nprobe = 16


class FaissPCAIndex(FaissTrainIndex):
    def __init__(self, d):
        d2 = 256
        nlist = 100 # numCentroids
        m = 8 # numQuantizers

        coarse_quantizer = faiss.IndexFlatL2(d2)
        sub_index = faiss.IndexIVFPQ(coarse_quantizer, d2, nlist, 16, 8)
        pca_matrix = faiss.PCAMatrix(d, d2, 0, True)
        self.index2 = faiss.IndexPreTransform(pca_matrix, sub_index)

        sub_index.own_fields = True
        coarse_quantizer.this.disown()

        self.sub_index = sub_index
        self.pca_matrix = pca_matrix

        self.index2.nprobe = 10

    
if __name__ == '__main__':
    import numpy as np
    d = 2048
    faiss_index = FaissIndex(d)
    nb = 10
    xb = np.random.random((nb, d)).astype('float32')
    ids = np.arange(nb) + 1
    faiss_index.add(xb, ids)
    print(faiss_index.ntotal())

    nq = 1
    xq = np.random.random((nq, d)).astype('float32')
    xq = xb[:1]
    D, I = faiss_index.search(xq)
    print(zip(I[0], D[0]))
