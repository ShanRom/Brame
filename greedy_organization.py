import numpy as np

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


class GreedyPartitioner(object):

    class Slice(object):
        def __init__(self, idx, vec, index):
            self.idx = idx
            self.vec = vec
            self.index = index.tolist()

    class Cluster(object):
        def __init__(self, idx, slicer):
            self.idx = idx
            self.pools = [slicer.idx]
            self.size = len(slicer.index)
            self.sketch = slicer.vec * self.size

        def get_feat(self):
            feat = self.sketch / self.size
            return feat

        def merge(self, other):
            self.pools += other.pools
            self.size += other.size
            self.sketch += other.sketch

    def __init__(self, args, data, workload, columns, index):
        self.args = args
        self.data = data
        self.workload = workload
        self.columns = columns
        self.index = index
        self.map = {idx: loc for loc, idx in enumerate(self.index)}

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

    def search(self, obj, clusters):
        obj_vec = obj.get_feat()
        max_dist = np.inf
        candidate = -1
        for cid in clusters.keys():
            if cid == obj.idx:
                continue
            else:
                sub = clusters[cid]
                sub_vec = sub.get_feat()
                dist = np.linalg.norm(obj_vec - sub_vec) * (sub.size / (sub.size + obj.size))
                if dist < max_dist:
                    max_dist = dist
                    candidate = cid
        return candidate

    def cluster(self, segments):
        clusters = {}
        for s in segments:
            cls = self.Cluster(idx=s, slicer=segments[s])
            clusters[s] = cls

        min_block_size = int(self.args.block_size * (1 - self.args.max_vacancy_ratio))
        heap = {s: clusters[s].size for s in clusters if clusters[s].size < min_block_size}

        while len(heap) > 0:
            obj_k = min(heap, key=heap.get)
            obj = clusters[obj_k]
            sub_k = self.search(obj=obj, clusters=clusters)
            clusters[sub_k].merge(clusters[obj_k])
            del heap[obj_k]
            del clusters[obj_k]
            if sub_k in heap:
                if clusters[sub_k].size >= min_block_size:
                    del heap[sub_k]
                else:
                    heap[sub_k] = clusters[sub_k].size

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

        clusters = self.cluster(segments=segments)

        blocks = self.block_generate(clusters=clusters, segments=segments)

        return blocks

