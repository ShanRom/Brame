import numpy as np
import pickle
import math


from constants import PKL_PROTO


class Node(object):
    def __init__(self, node_id, index, parent_id):
        self.node_id = node_id
        self.block_id = None

        self.index = index
        self.size = len(index)
        self.domains = None

        self.is_leaf = True
        self.can_split = True

        self.children = None
        self.parent = parent_id

    def set_children(self, children):
        self.children = children

    def set_domains(self, data):
        min_vals = data.min().values.reshape(-1, 1)
        max_vals = data.max().values.reshape(-1, 1)
        self.domains = np.concatenate((min_vals, max_vals), axis=1)

    def get_center(self):
        assert self.domains is not None
        center = np.mean(self.domains, axis=1)
        return center


def indexMatConstruct(blocks, columns, level='block'):
    index = {}
    lower_bound = np.zeros(shape=(len(blocks), len(columns)))
    upper_bound = np.zeros(shape=(len(blocks), len(columns)))
    for idx, b in enumerate(blocks.values()):
        if level == 'block':
            index[idx] = b.block_id
        elif level == 'node':
            index[idx] = b.node_id
        else:
            raise NotImplementedError
        p_dom_low = b.domains[:, 0]
        p_dom_up = b.domains[:, 1]
        lower_bound[idx] = p_dom_low
        upper_bound[idx] = p_dom_up
    Mapper = index
    indexMat = (lower_bound, upper_bound)
    return Mapper, indexMat


def fastDataLocate(q, columns, mapper, index_mat):
    l = q.generate_bitmap(index_mat[0], columns)
    u = q.generate_bitmap(index_mat[1], columns)
    ind = l | u
    blocks = np.where(ind == 1)[0]
    blocks = [mapper[b] for b in blocks]
    return set(blocks)


def getLeaves(tree, for_split=False):
    leaves = []
    if for_split:
        for node in tree.values():
            if node.is_leaf and node.can_split:
                leaves.append(node)
    else:
        for node in tree.values():
            if node.is_leaf:
                leaves.append(node)

    return leaves


def KeyOrdering(data, block_size):
    sorted_data = data.sort_values(by=list(data.columns), ascending=True)
    cursor = 0
    idx = 0
    chunks = {}
    while cursor < len(data) - block_size:
        chunk = sorted_data.iloc[cursor: cursor + block_size]
        chunks[idx] = chunk
        idx += 1
        cursor += block_size
    if cursor < len(data):
        chunk = sorted_data.iloc[cursor:]
        chunks[idx] = chunk

    return chunks


def HilbertFillingCurve(data, block_size, args):
    from hilbertcurve.hilbertcurve import HilbertCurve
    table = data.values
    table = np.round(table * (10 ** args.curve_order))
    order = math.ceil(math.log(10 ** args.curve_order + 1, 2))
    space_filling_curve = HilbertCurve(p=order, n=table.shape[1])
    dists = space_filling_curve.distances_from_points(points=table)
    ranks = np.argsort(dists)
    sorted_data = data.loc[ranks]
    cursor = 0
    idx = 0
    chunks = {}
    while cursor < len(data) - block_size:
        chunk = sorted_data.iloc[cursor: cursor + block_size]
        chunks[idx] = chunk
        idx += 1
        cursor += block_size
    if cursor < len(data):
        chunk = sorted_data.iloc[cursor:]
        chunks[idx] = chunk

    return chunks


def clusterWithCurve(args, sequence):

    class Cluster(object):
        def __init__(self, state):
            self.state = state
            self.items = []

        def add_item(self, item):
            self.items.append(item)

        def get_size(self):
            return len(self.items)

        def merge(self, other):
            self.items += other.items

    clusters = {}
    cls_idx = 0

    # initialize
    if sequence[0] < args.curve_filter_threshold:
        cluster = Cluster(state='cold')
    else:
        cluster = Cluster(state='warm')
    cluster.add_item(item=0)
    clusters[cls_idx] = cluster
    cls_idx += 1

    # update
    for cursor in range(1, len(sequence)):
        # deal with cold pages
        if sequence[cursor] < args.curve_filter_threshold:
            if clusters[cls_idx - 1].state == 'cold':
                clusters[cls_idx - 1].add_item(item=cursor)
            else:
                cluster = Cluster(state='cold')
                clusters[cls_idx] = cluster
                clusters[cls_idx].add_item(item=cursor)
                cls_idx += 1
        # deal with hot pages
        else:
            if clusters[cls_idx - 1].state == 'cold':
                cluster = Cluster(state='warm')
                clusters[cls_idx] = cluster
                clusters[cls_idx].add_item(item=cursor)
                cls_idx += 1
            else:
                clusters[cls_idx - 1].add_item(item=cursor)

    # combine hot zones
    cls_indexer = [k for k in clusters.keys()]
    cursor = 0
    shadow = -1
    while cursor < len(cls_indexer):
        cur = cls_indexer[cursor]
        if clusters[cur].state == 'warm':
            shadow = cursor
        else:
            if clusters[cur].get_size() > args.min_sequence_length:
                shadow = cursor
            else:
                if shadow == -1:
                    clusters[cur].state = 'warm'
                    succ = cls_indexer[cursor + 1]
                    clusters[cur].merge(clusters[succ])
                    del clusters[succ]
                    cursor += 1
                elif cursor == len(cls_indexer) - 1:
                    pred = cls_indexer[shadow]
                    clusters[pred].merge(clusters[cur])
                    del clusters[cur]
                else:
                    pred = cls_indexer[shadow]
                    succ = cls_indexer[cursor + 1]
                    clusters[pred].merge(clusters[cur])
                    clusters[pred].merge(clusters[succ])
                    del clusters[cur]
                    del clusters[succ]
                    cursor += 1
        cursor += 1

    hot_index = []
    cold_index = []
    for cid in clusters.keys():
        if clusters[cid].state == 'warm':
            hot_index += clusters[cid].items
        elif clusters[cid].state == 'cold':
            cold_index += clusters[cid].items
        else:
            raise NotImplementedError
    return hot_index, cold_index


def knapsackSolver(weights, values, capacity, algorithm='greedy'):
    if algorithm == 'dynamic_programming' or algorithm == 'dp':
        n = len(weights)

        dp = [[0] * (capacity + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(1, capacity + 1):
                if weights[i - 1] > j:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = max(dp[i - 1][j], values[i - 1] + dp[i - 1][j - weights[i - 1]])

        selected_items = []
        j = capacity
        for i in range(n, 0, -1):
            if dp[i][j] != dp[i - 1][j]:
                selected_items.append(i - 1)
                j -= weights[i - 1]

        return selected_items

    elif algorithm == 'greedy':
        backpack = [(i, values[i] / weights[i]) for i in range(len(weights))]
        backpack = sorted(backpack, key=lambda x: x[1], reverse=True)
        selected_items = []
        room = 0
        for item in backpack:
            index = item[0]
            if room + weights[index] > capacity:
                break
            room += weights[index]
            selected_items.append(index)

        return selected_items

    else:
        raise NotImplementedError


def save(obj, name, path):
    path.mkdir(parents=True, exist_ok=True)
    writer = open(path / f'{name}.pkl', 'wb')
    pickle.dump(obj, writer, protocol=PKL_PROTO)


def load(name, path):
    loader = open(path / f'{name}.pkl', 'rb')
    obj = pickle.load(loader)
    return obj


class Evaluator(object):
    def __init__(self, columns, blocks):
        self.columns = columns
        self.blocks = blocks

        mapper, index_mat = indexMatConstruct(blocks=blocks, columns=columns, level='block')

        self.blockIndexer = mapper
        self.blockIndexMat = index_mat

    def fast_query_locate(self, q):
        return fastDataLocate(q=q, columns=self.columns, mapper=self.blockIndexer, index_mat=self.blockIndexMat)

    def scan_size(self, q_blocks, cached_blocks):
        hit_blocks = q_blocks.intersection(cached_blocks)
        miss_blocks = q_blocks.difference(cached_blocks)

        hit_tuple_size = sum([self.blocks[b].size for b in hit_blocks])
        miss_tuple_size = sum([self.blocks[b].size for b in miss_blocks])
        return hit_tuple_size, miss_tuple_size

    def cache_hit(self, q, q_blocks, cached_blocks, level):
        hit_blocks = q_blocks.intersection(cached_blocks)
        miss_blocks = q_blocks.difference(cached_blocks)

        if level == 'block':
            return len(hit_blocks), len(miss_blocks)

        elif level == 'tuple':
            hit_tuple_size = 0
            for b_id in hit_blocks:
                block = self.blocks[b_id]
                bitmap = q.generate_bitmap(sample=block.data.values, col2idx=self.columns)
                hit_tuple_size += np.sum(bitmap)

            miss_tuple_size = 0
            for b_id in miss_blocks:
                block = self.blocks[b_id]
                bitmap = q.generate_bitmap(sample=block.data.values, col2idx=self.columns)
                miss_tuple_size += np.sum(bitmap)

            return hit_tuple_size, miss_tuple_size

        else:
            raise NotImplementedError
