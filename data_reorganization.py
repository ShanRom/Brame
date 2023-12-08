import numpy as np
import math
import time
import tqdm
from collections import defaultdict

from utility import Node
from cluster import cluster
from table_partition import PartitionTree, PartitionTreePlus
from partition_organization import Partitioner
from greedy_organization import GreedyPartitioner


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


def KDTreePlus(data, index, block_size, root_id, layer):
    nodes = PartitionTreePlus(data=data, index=index, block_size=block_size, layer=layer)
    blocks = []
    offset = 1
    for node in nodes:
        node.node_id = root_id + offset
        node.set_domains(data=data.loc[node.index])
        offset += 1
        blocks.append(node)
    return blocks, root_id + offset


class HBC(object):
    def __init__(self, args, data, workload, columns, index):
        self.args = args
        self.data = data
        self.workload = workload
        self.columns = columns
        self.index = index
        self.map = {idx: loc for loc, idx in enumerate(self.index)}

    def workload_filter(self):
        min_vals = self.data.min().values.reshape(-1, 1)
        max_vals = self.data.max().values.reshape(-1, 1)
        f_box = np.concatenate((min_vals, max_vals), axis=1)

        workload = []
        for q in self.workload:
            q_dom = q.domains_extract()
            q_box = np.zeros(shape=(len(self.columns), 2))
            q_box[:, 1] = 1.0
            for c in q_dom.keys():
                q_box[self.columns[c]][0], q_box[self.columns[c]][1] = q_dom[c][0], q_dom[c][1]

                box_1 = f_box[:, 0] - q_box[:, 1]
                box_2 = q_box[:, 0] - f_box[:, 1]
                if np.max(box_1) > 0 or np.max(box_2) > 0:
                    continue
                else:
                    workload.append(q)
        self.workload = workload

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

    def cluster(self, data, K):
        index = cluster(data=data, K=K, args=self.args, algorithm=self.args.cluster_method)
        return index

    def hierarchical_cluster(self, matrix, index=None):

        if index is None:
            index = self.index

        Count = 0

        leaves = {0: Node(node_id=0, index=index, parent_id=-1)}

        CanSplit = True

        while CanSplit:
            CanSplit = False

            leaves_for_split = [leaf for leaf in leaves.values() if leaf.can_split]

            if len(leaves) == 0:
                break

            for leaf in leaves_for_split:

                if leaf.size < 2 * self.args.block_size:
                    leaves[leaf.node_id].can_split = False
                    continue

                if leaf.size >= self.args.K * self.args.block_size:
                    K = self.args.K
                elif leaf.size >= int(math.sqrt(self.args.K)) * self.args.block_size:
                    K = int(math.sqrt(self.args.K))
                else:
                    K = 2

                row_index = [self.map[idx] for idx in leaf.index]
                mat = matrix[row_index, :]

                uni_vectors = np.unique(mat, axis=0)

                if len(uni_vectors) <= K:

                    nodes, Count = KDTree(data=self.data,
                                          index=leaf.index,
                                          block_size=self.args.block_size,
                                          root_id=Count)

                    for node in nodes:
                        leaves[node.node_id] = node

                else:
                    index = self.cluster(data=mat, K=K)

                    container = defaultdict(list)
                    for idx in range(len(index)):
                        container[index[idx]].append(leaf.index[idx])

                    min_cluster_size = min([len(container[k]) for k in container.keys()])
                    if min_cluster_size < self.args.block_size / 2:

                        nodes, Count = KDTree(data=self.data,
                                              index=leaf.index,
                                              block_size=self.args.block_size,
                                              root_id=Count)

                        for node in nodes:
                            leaves[node.node_id] = node

                    else:
                        for label in container.keys():
                            Count += 1
                            node = Node(node_id=Count, index=container[label], parent_id=leaf.node_id)
                            node.set_domains(data=self.data.loc[node.index])
                            leaves[Count] = node

                leaves.pop(leaf.node_id)

                CanSplit = True

        return leaves

    def hierarchical_cluster_plus(self, matrix, layer, index=None):

        if index is None:
            index = self.index
        Count = 0
        Layer = 0

        leaves = {0: Node(node_id=0, index=index, parent_id=-1)}

        CanSplit = True

        while CanSplit and Layer < layer:
            CanSplit = False

            leaves_for_split = [leaf for leaf in leaves.values() if leaf.can_split]

            if len(leaves) == 0:
                break

            for leaf in leaves_for_split:

                Layer += 1

                if leaf.size < 2 * self.args.block_size:
                    leaves[leaf.node_id].can_split = False
                    continue

                if leaf.size >= self.args.K * self.args.block_size:
                    K = self.args.K
                elif leaf.size >= int(math.sqrt(self.args.K)) * self.args.block_size:
                    K = int(math.sqrt(self.args.K))
                else:
                    K = 2

                row_index = [self.map[idx] for idx in leaf.index]
                mat = matrix[row_index, :]

                uni_vectors = np.unique(mat, axis=0)

                if len(uni_vectors) <= K:

                    n_layer = int(math.log(K, 2))

                    nodes, Count = KDTreePlus(data=self.data,
                                              index=leaf.index,
                                              block_size=self.args.block_size,
                                              root_id=Count,
                                              layer=n_layer)

                    for node in nodes:
                        leaves[node.node_id] = node

                else:
                    index = self.cluster(data=mat, K=K)

                    container = defaultdict(list)
                    for idx in range(len(index)):
                        container[index[idx]].append(leaf.index[idx])

                    min_cluster_size = min([len(container[k]) for k in container.keys()])
                    if min_cluster_size < self.args.block_size / 2:

                        n_layer = int(math.log(K, 2))
                        nodes, Count = KDTreePlus(data=self.data,
                                                  index=leaf.index,
                                                  block_size=self.args.block_size,
                                                  root_id=Count,
                                                  layer=n_layer)

                        for node in nodes:
                            leaves[node.node_id] = node

                    else:
                        for label in container.keys():
                            Count += 1
                            node = Node(node_id=Count, index=container[label], parent_id=leaf.node_id)
                            node.set_domains(data=self.data.loc[node.index])
                            leaves[Count] = node

                leaves.pop(leaf.node_id)

                CanSplit = True

        return leaves

    def hierarchical_cluster_with_indexer(self, matrix, index=None):
        if index is None:
            index = self.index

        num_leaves = len(index) / self.args.block_size
        num_scan_segment = max(1, int(num_leaves / self.args.max_scan_blocks_num))
        if len(index) >= self.args.K * self.args.block_size:
            K = self.args.K
        elif len(index) >= int(math.sqrt(self.args.K)) * self.args.block_size:
            K = int(math.sqrt(self.args.K))
        else:
            K = 2
        Layer = int(math.log(num_scan_segment, K))
        if Layer > 1:
            parts = self.hierarchical_cluster_plus(matrix, Layer)
            roots = []
            forest = []
            for part in parts.values():
                roots.append(part.domains)
                leaves = self.hierarchical_cluster(matrix, part.index)
                forest.append(leaves)
        else:
            leaves = self.hierarchical_cluster(matrix)
            root = Node(node_id=0, index=self.index, parent_id=-1)
            root.set_domains(data=self.data.loc[self.index])
            roots = [root]
            forest = [leaves]
        return forest, roots

    def pipeline(self, with_indexer=False):
        self.workload_filter()
        matrix = self.tuple_encode(with_compress=True)
        if not with_indexer:
            blocks = self.hierarchical_cluster(matrix)
            return blocks
        else:
            hot_forest, hot_indexer = self.hierarchical_cluster_with_indexer(matrix)
            return hot_forest, hot_indexer


class PartitionOrganizer(object):
    def __init__(self, args, data, workload, columns, partitions):
        self.args = args
        self.data = data
        self.workload = workload
        self.columns = columns
        self.partitions = partitions

    def pipeline(self, algorithm):
        forest = {}

        if algorithm == 'pb_hbc' or algorithm == 'qb_hbc':
            for pid in self.partitions:
                blocks = HBC(args=self.args, data=self.data, workload=self.workload, columns=self.columns,
                             index=self.partitions[pid]).pipeline()
                forest[pid] = blocks

        elif algorithm == 'pb_mbm':
            for pid in self.partitions:
                blocks = Partitioner(args=self.args, data=self.data, workload=self.workload, columns=self.columns,
                                     index=self.partitions[pid]).pipeline()
                forest[pid] = blocks

        elif algorithm == 'pb_gbm':
            for pid in self.partitions:
                blocks = GreedyPartitioner(args=self.args, data=self.data, workload=self.workload, columns=self.columns,
                                           index=self.partitions[pid]).pipeline()
                forest[pid] = blocks

        else:
            raise NotImplementedError

        return forest
