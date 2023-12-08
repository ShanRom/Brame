import math
import time
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import copy

from utility import Node
from cluster import cluster, hierarchical_cluster


def PartitionTree(data, index, block_size):
    Count = 0

    leaves = {Count: Node(node_id=Count, index=index, parent_id=-1)}

    CanSplit = True

    while CanSplit:
        CanSplit = False

        leaves_for_split = [leaf for leaf in leaves.values() if leaf.can_split]

        if len(leaves_for_split) == 0:
            break

        for leaf in leaves_for_split:

            if leaf.size < 2 * block_size:
                leaves[leaf.node_id].can_split = False
                continue

            leaf_data = data.loc[leaf.index]
            vars = leaf_data.var()
            split_col = vars.idxmax()
            median = leaf_data[split_col].median()

            left_loc = np.where(leaf_data[split_col] <= median)[0]
            right_loc = np.where(leaf_data[split_col] > median)[0]

            if len(left_loc) < block_size / 4 or len(right_loc) < block_size / 4:
                K = math.ceil(leaf.size / block_size)
                sizes = [block_size for _ in range(K)]
                sizes[-1] = leaf.size - (K - 1) * block_size
                chunks = []
                offset = 0

                for idx in range(K):
                    if idx != K - 1:
                        chunk = leaf.index[offset: offset + sizes[idx]]
                    else:
                        chunk = leaf.index[offset:]
                    chunks.append(chunk)
                    offset += sizes[idx]

                for idx in range(K):
                    node = Node(node_id=Count + idx + 1, index=chunks[idx], parent_id=leaf.node_id)
                    node.can_split = False
                    leaves[Count + idx + 1] = copy.deepcopy(node)

                leaves.pop(leaf.node_id)
                Count += K
                continue

            left_index = [leaf.index[i] for i in left_loc]
            right_index = [leaf.index[i] for i in right_loc]

            leaves.pop(leaf.node_id)

            left_child = Node(node_id=Count + 1, index=left_index, parent_id=leaf.node_id)
            leaves[left_child.node_id] = copy.deepcopy(left_child)

            right_child = Node(node_id=Count + 2, index=right_index, parent_id=leaf.node_id)
            leaves[right_child.node_id] = copy.deepcopy(right_child)

            Count += 2
            CanSplit = True

    leaves = [leaf for leaf in leaves.values()]
    for idx in range(len(leaves)):
        leaves[idx].set_domains(data=data.loc[leaves[idx].index])

    return leaves


def PartitionTreePlus(data, index, block_size, layer):
    Count = 0
    Layer = 0
    leaves = {Count: Node(node_id=Count, index=index, parent_id=-1)}

    CanSplit = True

    while CanSplit and Layer < layer:
        CanSplit = False

        leaves_for_split = [leaf for leaf in leaves.values() if leaf.can_split]

        if len(leaves_for_split) == 0:
            break

        for leaf in leaves_for_split:

            Layer += 1

            if leaf.size < 2 * block_size:
                leaves[leaf.node_id].can_split = False
                continue

            leaf_data = data.loc[leaf.index]
            vars = leaf_data.var()
            split_col = vars.idxmax()
            median = leaf_data[split_col].median()

            left_loc = np.where(leaf_data[split_col] <= median)[0]
            right_loc = np.where(leaf_data[split_col] > median)[0]

            if len(left_loc) < block_size / 4 or len(right_loc) < block_size / 4:
                split_point = int(leaf_data / 2)

                left_chunk = leaf.index[:split_point]
                right_chunk = leaf.index[split_point:]
                left_child = Node(node_id=Count + 1, index=left_chunk, parent_id=leaf.node_id)
                right_child = Node(node_id=Count + 2, index=right_chunk, parent_id=leaf.node_id)
                leaves[Count + 1] = left_child
                leaves[Count + 2] = right_child

                leaves.pop(leaf.node_id)
                Count += 2
                continue

            left_index = [leaf.index[i] for i in left_loc]
            right_index = [leaf.index[i] for i in right_loc]

            leaves.pop(leaf.node_id)

            left_child = Node(node_id=Count + 1, index=left_index, parent_id=leaf.node_id)
            leaves[left_child.node_id] = copy.deepcopy(left_child)

            right_child = Node(node_id=Count + 2, index=right_index, parent_id=leaf.node_id)
            leaves[right_child.node_id] = copy.deepcopy(right_child)

            Count += 2
            CanSplit = True

    leaves = [leaf for leaf in leaves.values()]
    for idx in range(len(leaves)):
        leaves[idx].set_domains(data=data.loc[leaves[idx].index])

    return leaves


def PartitionTreeWithIndexTree(data, index, block_size, scan_size):
    num_leaves = len(index) / block_size
    num_scan_segment = int(num_leaves / scan_size)
    if num_scan_segment > 1:
        Layer = int(math.log(num_scan_segment, 2))
        parts = PartitionTreePlus(data, index, block_size, layer=Layer)
        roots = []
        forest = []
        for part in parts:
            roots.append(part.domains)
            leaves = PartitionTree(data, part.index, block_size)
            forest.append(leaves)
    else:
        leaves = PartitionTree(data, index, block_size)
        root = Node(node_id=0, index=index, parent_id=-1)
        root.set_domains(data=data.loc[index])
        roots = [root.domains]
        forest = [leaves]
    return forest, roots


class TablePartitioner(object):
    def __init__(self, args, data, workload, columns):
        self.args = args
        self.data = data
        self.workload = workload
        self.columns = columns

    def pre_partition(self):
        self.data = self.data.sort_values(by=list(self.data.columns), ascending=True)
        index = [i for i in range(len(self.data))]
        pages = PartitionTree(data=self.data, index=index, block_size=self.args.page_size)
        return pages

    def page_order(self, pages, order_method):
        pages = {p.node_id: p for p in pages}

        if order_method == 'hilbert_curve':
            from hilbertcurve.hilbertcurve import HilbertCurve

            pageIndex = {p: idx for idx, p in enumerate(pages)}
            indexPage = {v: k for k, v in pageIndex.items()}
            matrix = np.zeros(shape=(len(pages), len(self.columns)))
            for pid in pages:
                center = pages[pid].get_center()
                center = np.round(center * (10 ** self.args.curve_order))
                matrix[pageIndex[pid]] = center

            order = math.ceil(math.log(10 ** self.args.curve_order + 1, 2))
            space_filling_curve = HilbertCurve(p=order, n=len(self.columns))
            locations = space_filling_curve.distances_from_points(points=matrix)
            ranks = np.argsort(locations)
            sequence = [-1 for _ in range(len(pages))]
            for i, r in enumerate(ranks):
                sequence[r] = indexPage[i]

            ordered_pages = []
            for rank, pid in enumerate(sequence):
                page = pages[pid]
                page.node_id = rank
                ordered_pages.append(page)
            return ordered_pages

        elif order_method == 'key_order':
            pageIndex = {p: idx for idx, p in enumerate(pages)}
            indexPage = {v: k for k, v in pageIndex.items()}
            matrix = np.zeros(shape=(len(pages), len(self.columns)))
            for pid in pages:
                center = pages[pid].get_center()
                matrix[pageIndex[pid]] = center

            df = pd.DataFrame(matrix, columns=self.data.columns)
            index = df.sort_values(by=list(df.columns), ascending=True).index.values
            sequence = [indexPage[i] for i in index]

            ordered_pages = []
            for rank, pid in enumerate(sequence):
                page = pages[pid]
                page.node_id = rank
                ordered_pages.append(page)
            return ordered_pages

        else:
            raise NotImplementedError

    def fast_page_encode(self, pages, index):
        from utility import fastDataLocate, indexMatConstruct
        mat = np.zeros(shape=(len(pages), len(self.workload)))
        mapper, index_mat = indexMatConstruct(blocks=pages, columns=self.columns, level='node')
        for i, q in enumerate(self.workload):
            routed_pages = fastDataLocate(q=q, columns=self.columns, mapper=mapper, index_mat=index_mat)
            routed_pages = [index[p] for p in routed_pages]
            mat[routed_pages, i] = 1
        return mat

    def page_filter(self, mat):
        freq_mat = np.sum(mat, axis=1)
        from utility import clusterWithCurve
        hot_index, cold_index = clusterWithCurve(args=self.args, sequence=freq_mat)
        return hot_index, cold_index

    def space_filling_curve(self, pages):
        from curve import SpaceFillingCurve
        agent = SpaceFillingCurve(args=self.args, pages=pages, data=self.data,
                                  workload=self.workload, columns=self.columns)
        clusters = agent.pipeline()
        return clusters

    def cluster(self, data, K):
        index = cluster(data=data, K=K, args=self.args, algorithm=self.args.partition_generate_strategy)
        return index

    def cold_block_generate(self, method, indexer):
        if method == 'kd_tree':
            cold_blocks = PartitionTree(data=self.data, index=indexer, block_size=self.args.block_size)
        elif method == 'key_order' or method == 'hilbert_curve':
            indexer = sorted(indexer)
            cold_blocks = {}
            cursor = 0
            idx = 0
            while cursor < len(indexer) - self.args.block_size:
                index = indexer[cursor: cursor + self.args.block_size]
                block = Node(node_id=idx, index=index, parent_id=-1)
                block.set_domains(data=self.data.loc[index])
                cold_blocks[idx] = block
                idx += 1
                cursor += self.args.page_size
            if cursor < len(self.data):
                index = [i for i in range(cursor, len(self.data))]
                block = Node(node_id=idx, index=index, parent_id=-1)
                block.set_domains(data=self.data.loc[index])
                cold_blocks[idx] = block
            cold_blocks = [b for b in cold_blocks.values()]
        else:
            raise NotImplementedError
        return cold_blocks

    def cold_block_generate_with_indexer(self, method, indexer):
        assert method == 'kd_tree'
        cold_forest, roots = PartitionTreeWithIndexTree(data=self.data, index=indexer, block_size=self.args.block_size,
                                                        scan_size=self.args.max_scan_blocks_num)
        return cold_forest, roots

    def pipeline(self, with_indexer=False):
        since = time.time()
        pages = self.pre_partition()
        print('[Pre-Partition] ', self.args.page_generate_strategy, ' Time Consume : ', time.time() - since)

        pages = self.page_order(pages=pages, order_method=self.args.page_order)

        page2index = {p.node_id: i for i, p in enumerate(pages)}
        index2page = {v: k for k, v in page2index.items()}
        pages_dictionary = {p.node_id: p for p in pages}

        if self.args.partition_generate_strategy == 'curve':
            since = time.time()
            clusters = self.space_filling_curve(pages=pages_dictionary)
            print('[Page Cluster With Hilbert Curve] Time Consume : ', time.time() - since)
            partitions = defaultdict(list)
            cold_tuple_index = []
            pid = 0
            for cls in clusters.values():
                if cls.state == 'warm':
                    for item in cls.items:
                        partitions[pid] += pages_dictionary[item].index
                    pid += 1
                else:
                    for item in cls.items:
                        cold_tuple_index += pages_dictionary[item].index
            since = time.time()
            cold_blocks = self.cold_block_generate(method=self.args.cold_block_generator, indexer=cold_tuple_index)
            print('[Cold Pages Index Construction] : ', time.time() - since)
            return partitions, cold_blocks

        since = time.time()
        matrix = self.fast_page_encode(pages=pages_dictionary, index=page2index)
        print('[Page Encoding] Time Consume : ', time.time() - since)

        hot_index, cold_index = self.page_filter(mat=matrix)

        since = time.time()
        matrix = matrix[hot_index, :]
        partitions = {}

        assign_arr = self.cluster(data=matrix, K=self.args.num_cluster)

        if self.args.post_partition:
            monitor = Counter(assign_arr)
            for c in monitor.keys():
                partitions[c] = {}
                if monitor[c] * self.args.page_size > self.args.max_cluster_scale * len(self.data):
                    inds = np.where(assign_arr == c)[0]
                    tmp = np.zeros(shape=(len(inds), len(self.columns)))
                    for loc, i in enumerate(inds):
                        center = pages[hot_index[i]].get_center()
                        tmp[loc] = center
                    sub_clusters = hierarchical_cluster(args=self.args,
                                                        data=pd.DataFrame(matrix[inds, :]),
                                                        matrix=pd.DataFrame(tmp),
                                                        n_cluster=2,
                                                        cluster_size=self.args.max_cluster_scale * len(
                                                            self.data) / self.args.page_size,
                                                        algorithm=self.args.cluster_method)
                    for i, cls in enumerate(sub_clusters):
                        indexer = np.where(assign_arr == c)[0][cls]
                        partitions[c][i] = []
                        for idx in indexer:
                            partitions[c][i] += pages_dictionary[index2page[hot_index[idx]]].index
                else:
                    indexer = np.where(assign_arr == c)[0]
                    partitions[c][0] = []
                    for idx in indexer:
                        partitions[c][0] += pages_dictionary[index2page[hot_index[idx]]].index

            counter = 0
            processed_partitions = {}
            for pid in partitions.keys():
                for cid in partitions[pid].keys():
                    processed_partitions[counter] = partitions[pid][cid]
                    counter += 1
            partitions = processed_partitions

        else:
            partitions = defaultdict(list)

            for i in range(len(assign_arr)):
                partitions[assign_arr[i]] += pages_dictionary[index2page[hot_index[i]]].index

            counter = 0
            processed_partitions = {}
            for part in partitions.values():
                if len(part) == 0:
                    continue
                else:
                    processed_partitions[counter] = part
                    counter += 1
            partitions = processed_partitions

        print('[Hot Pages Cluster] ', self.args.partition_generate_strategy, ' Time Consume : ', time.time() - since)

        since = time.time()
        cold_tuple_index = []
        for idx in cold_index:
            cold_tuple_index += pages_dictionary[index2page[idx]].index

        if not with_indexer:
            cold_blocks = self.cold_block_generate(method=self.args.cold_block_generator, indexer=cold_tuple_index)
            print('[Cold Pages Index Construction] : ', time.time() - since)

            return partitions, cold_blocks

        else:
            cold_forest, cold_indexer = self.cold_block_generate_with_indexer(method=self.args.cold_block_generator,
                                                                              indexer=cold_tuple_index)
            print('[Cold Pages Index Construction] : ', time.time() - since)

            return partitions, cold_forest, cold_indexer
