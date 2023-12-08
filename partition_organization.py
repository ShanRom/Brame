import numpy as np
import pandas as pd
import math
import time
import tqdm
from collections import defaultdict, Counter
from datasketch import MinHash, MinHashLSH

from utility import Node
from table_partition import PartitionTree


def KDTree(data, index, block_size, root_id):
    nodes = PartitionTree(data=data, index=index, block_size=block_size)
    blocks = []
    offset = 1
    for node in nodes:
        node.node_id = root_id + offset
        node.can_split = False
        node.set_domains(data=data.loc[node.index])
        offset += 1
        blocks.append(node)

    return blocks, root_id + offset


class Partitioner(object):

    class Slice(object):
        def __init__(self, idx, vec, index):
            self.idx = idx
            self.vec = vec
            self.index = index.tolist()
            self.feat = [str(i) for i in np.where(self.vec == 1)[0]]

    class Cluster(object):
        def __init__(self, idx, slicer):
            self.idx = idx
            self.pools = [slicer.idx]
            self.size = len(slicer.index)
            self.sketch = defaultdict(int)
            for k in slicer.feat:
                self.sketch[k] = self.size

        def get_feat(self, threshold):
            feat = [k for k in self.sketch if self.sketch[k] / self.size > threshold]
            return feat

        def merge(self, other):
            self.pools += other.pools
            self.size += other.size
            for k in other.sketch:
                self.sketch[k] += other.sketch[k]

    def __init__(self, args, data, workload, columns, index):
        self.args = args
        self.data = data
        self.workload = workload
        self.columns = columns
        self.index = index
        self.map = {idx: loc for loc, idx in enumerate(self.index)}

        num_perm = args.hash_K * args.hash_L
        self.lsh = MinHashLSH(num_perm=num_perm, threshold=args.lsh_threshold, params=(args.hash_K, args.hash_L))

    def tuple_encode(self, with_compress=False):
        data = self.data.loc[self.index]
        mat = np.zeros(shape=(len(self.index), len(self.workload)))
        column_for_compress = []
        for i, q in enumerate(self.workload):
            bitmap = q.generate_bitmap(sample=data.values, col2idx=self.columns)
            if int(np.sum(bitmap)) == 0 or int(np.sum(bitmap)) == len(bitmap):
                column_for_compress.append(i)
            mat[:, i] = bitmap

        if with_compress:
            columns = list(set([i for i in range(len(self.workload))]).difference(column_for_compress))
            mat = mat[:, columns] if len(columns) > 0 else mat

        return mat

    def index_construct(self, mat):
        segments = {}
        uni_mat = np.unique(mat, axis=0)
        print(uni_mat.shape)
        for i in range(len(uni_mat)):
            vec = uni_mat[i]
            index = np.where(np.all(mat == vec, axis=1))[0]
            cls = self.Slice(idx=i, vec=vec, index=index)
            segments[i] = cls
        return segments

    def lsh_construct(self, segments):
        for s in segments.values():
            feat = s.feat
            minhash = MinHash(num_perm=self.args.hash_K * self.args.hash_L)
            minhash.update_batch(p.encode('utf-8') for p in feat)
            self.lsh.insert(str(s.idx), minhash)

    def lsh_query(self, feat):
        minhash = MinHash(num_perm=self.args.hash_K * self.args.hash_L)
        minhash.update_batch(p.encode('utf-8') for p in feat)
        neighbors = self.lsh.query(minhash)
        if len(neighbors) > 0:
            neighbors = [int(i) for i in neighbors]
        return neighbors

    def cluster(self, segments, cls_threshold):
        mapper = {}
        clusters = {}
        for s in segments:
            cls = self.Cluster(idx=s, slicer=segments[s])
            clusters[s] = cls
            mapper[s] = s

        min_block_size = int(self.args.block_size * (1 - self.args.max_vacancy_ratio))
        heap = {s: clusters[s].size for s in clusters if clusters[s].size < min_block_size}
        outliers = []

        while len(heap) > 0:
            obj_k = min(heap, key=heap.get)
            obj = clusters[obj_k]
            feat = obj.get_feat(threshold=cls_threshold)
            neighbors = self.lsh_query(feat=feat)
            if len(neighbors) == 0:
                outliers.append(obj_k)
                del heap[obj_k]
            else:
                neighbors = [mapper[s] for s in neighbors]
                counter = Counter(neighbors)
                del counter[obj_k]
                if len(counter) == 0:
                    outliers.append(obj_k)
                    del heap[obj_k]
                else:
                    voting = {k: v * v / clusters[k].size for k, v in counter.items()}
                    sub_k = max(voting, key=voting.get)
                    clusters[sub_k].merge(obj)
                    for c in obj.pools:
                        mapper[c] = sub_k
                    del clusters[obj_k]
                    del heap[obj_k]
                    if sub_k in heap:
                        if clusters[sub_k].size >= min_block_size:
                            del heap[sub_k]
                        else:
                            heap[sub_k] = clusters[sub_k].size

        if len(outliers) > 1:
            sub_k = outliers[0]
            for i in range(1, len(outliers)):
                obj_k = outliers[i]
                clusters[sub_k].merge(clusters[obj_k])
                for c in clusters[obj_k].pools:
                    mapper[c] = sub_k
                del clusters[obj_k]

        return clusters

    def block_generate(self, clusters, segments):

        Count = 0

        leaves = {}

        for cid, c in clusters.items():
            index = []
            for sid in c.pools:
                index += segments[sid].index
            nodes, Count = KDTree(data=self.data, index=index, block_size=self.args.block_size, root_id=Count)
            for node in nodes:
                leaves[node.node_id] = node

        return leaves

    def pipeline(self):

        matrix = self.tuple_encode(with_compress=False)

        segments = self.index_construct(mat=matrix)

        self.lsh_construct(segments=segments)

        clusters = self.cluster(segments=segments, cls_threshold=self.args.lsh_filter_threshold)

        blocks = self.block_generate(clusters=clusters, segments=segments)

        return blocks

