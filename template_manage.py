import numpy as np
import copy
from collections import Counter
from datasketch import MinHash, MinHashLSH

from utility import indexMatConstruct, fastDataLocate, knapsackSolver


class Template(object):
    def __init__(self, template_id, q):
        self.template_id = template_id
        self.blocks = None
        self.workload = [q]
        self.hits = 1
        self.sketch = {}

    def add_hits(self):
        self.hits += 1

    def update(self, q_id, q, blocks, partitions, columns):
        self.workload.append(q_id)
        for b in blocks:
            if b not in self.sketch:
                self.sketch[b] = 1
            else:
                self.sketch[b] += 1

    def get_weighted_blocks(self):
        blocks = {b: self.sketch[b] / len(self.workload) for b in self.sketch}
        return blocks


class Cluster(object):
    def __init__(self, args, blocks, columns):
        self.args = args
        self.columns = columns
        self.blocks = blocks
        self.num_perm = self.args.hash_K * self.args.hash_L
        self.lsh = MinHashLSH(num_perm=self.num_perm, threshold=self.args.lsh_threshold,
                              params=(self.args.hash_K, self.args.hash_L))

        self.templates = []
        self.map = {}

        mapper, index_mat = indexMatConstruct(blocks=blocks, columns=columns, level='block')

        self.blockIndexer = mapper
        self.blockIndexMat = index_mat

    def encode(self, feature):
        minhash = MinHash(num_perm=self.num_perm)
        minhash.update_batch(p.encode('utf-8') for p in feature)
        return minhash

    def query_route(self, q):
        return fastDataLocate(q=q, columns=self.columns, mapper=self.blockIndexer, index_mat=self.blockIndexMat)

    def update(self, idx, feature):
        minhash = self.encode(feature=feature)
        self.lsh.insert(str(idx), minhash)

    def clear(self):
        for i in range(len(self.templates)):
            self.templates[i].hits = 0

    def cluster(self, workload_batch):
        for idx, query in workload_batch.items():

            blocks = self.query_route(q=query)

            feature = sorted([str(b) for b in blocks])
            minhash = self.encode(feature)
            neighbors = self.lsh.query(minhash)

            if len(neighbors) == 0:
                template = Template(template_id=len(self.templates), q=idx)
                template.update(q_id=idx, q=query, blocks=blocks, partitions=self.blocks, columns=self.columns)
                assign_template_idx = template.template_id
                self.templates.append(template)

            else:
                neighbors = [self.map[i] for i in neighbors]
                counter = Counter(neighbors)
                voting = {k: v * v / len(self.templates[k].workload) for k, v in counter.items()}
                assign_template_idx = max(voting, key=voting.get)
                self.templates[assign_template_idx].add_hits()
                self.templates[assign_template_idx].update(q_id=idx, q=query, blocks=blocks,
                                                           partitions=self.blocks, columns=self.columns)

            self.update(idx, feature)
            self.map[str(idx)] = assign_template_idx


class Listener(object):
    def __init__(self, block):
        self.block_id = block.block_id
        self.size = len(block.data)
        self.domains = block.domains

        self.t_heat = 0.0
        self.b_heat = 0.0
        self.t_hits = 0
        self.b_hits = 0

    def compute_temperature(self, forget_ratio):

        self.t_heat = self.t_heat * (1 - forget_ratio) + self.t_hits * forget_ratio
        self.b_heat = self.b_heat * (1 - forget_ratio) + self.b_hits * forget_ratio

        self.t_hits = 0
        self.b_hits = 0

    def set_hits(self, hits, level):
        if level == 'block':
            self.b_hits = hits
        elif level == 'template':
            self.t_hits = hits
        else:
            raise NotImplementedError


class Placer(object):
    def __init__(self, args, blocks, columns):
        self.args = args

        self.columns = columns
        mapper, index_mat = indexMatConstruct(blocks=blocks, columns=columns, level='block')

        self.blockIndexer = mapper
        self.blockIndexMat = index_mat

        segments = {}
        for bid in blocks.keys():
            listener = Listener(blocks[bid])
            segments[bid] = listener
        self.blocks = segments
        self.partitions = blocks

        self.block_in_cache = []
        self.block_in_cloud = []

        self.map = {k: [] for k in self.blocks.keys()}

    def index(self, templates):
        self.map = {k: [] for k in self.blocks.keys()}

        for t in templates:
            blocks = t.get_weighted_blocks()
            for b, w in blocks.items():
                self.map[b].append((t.template_id, w))

    def query_route(self, q):
        return fastDataLocate(q=q, columns=self.columns, mapper=self.blockIndexer, index_mat=self.blockIndexMat)

    def update(self, batch):
        for q in batch.values():
            blocks = self.query_route(q=q)
            for bid in blocks:
                self.blocks[bid].b_hits += 1

    def heat_compute(self, templates):
        for b in self.blocks.keys():
            corr_templates = self.map[b]

            hits = 0

            for t in corr_templates:
                hits += templates[t[0]].hits * t[1]
            self.blocks[b].set_hits(hits=hits, level='template')
            self.blocks[b].compute_temperature(forget_ratio=self.args.forget_ratio)

    def block_place(self):
        weights = [b.size for b in self.blocks.values()]

        values = [b.b_heat * self.args.heat_weight + b.t_heat * (1-self.args.heat_weight) for b in self.blocks.values()]

        block_in_cache = knapsackSolver(weights=weights, values=values, capacity=self.args.cache_budget,
                                        algorithm=self.args.place_strategy)
        block_in_cloud = list(set([i for i in range(len(self.blocks))]).difference(set(block_in_cache)))

        index = [b for b in self.blocks.keys()]
        self.block_in_cache = [index[i] for i in block_in_cache]
        self.block_in_cloud = [index[i] for i in block_in_cloud]


class TemplateScheduler(object):
    def __init__(self, args, columns, blocks):
        self.args = args
        self.columns = columns

        self.monitor = Cluster(args=args, columns=self.columns, blocks=blocks)
        self.scheduler = Placer(args=args, blocks=blocks, columns=columns)

    def supervise(self, workload_batch):
        self.monitor.cluster(workload_batch=workload_batch)
        self.scheduler.update(batch=workload_batch)

    def migrate(self):
        templates = self.monitor.templates
        self.scheduler.index(templates=templates)
        self.scheduler.heat_compute(templates=templates)
        self.scheduler.block_place()
        self.monitor.clear()

    def get_blocks(self):
        return self.scheduler.block_in_cache, self.scheduler.block_in_cloud

