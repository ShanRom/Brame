"""
Simulate Cloud-edge-end Cache Replacement
"""

import argparse
import random
import time
import pandas as pd
import numpy as np
import math

from constants import DATA_ROOT
from structures import load
from block_generate import BlockGenerator
from block_manage import BlockScheduler
from cache_replace_strategy import get_model
from utility import indexMatConstruct, fastDataLocate
from utility import Evaluator


class Cloud(object):
    def __init__(self):
        self.block_in_cloud = None

    def migrate(self, blocks):
        self.block_in_cloud = blocks

    def store(self):
        return self.block_in_cloud


class Edge(object):
    def __init__(self, args, columns, blocks):
        self.args = args
        self.scheduler = BlockScheduler(args, blocks, columns)
        self.block_in_edge = None

    def monitor(self, batch):
        self.scheduler.supervise(batch)

    def migrate(self):
        self.scheduler.migrate(heat_model=self.args.heat_model)
        block_in_edge, block_in_cloud = self.scheduler.get_blocks()
        self.block_in_edge = block_in_edge
        return block_in_cloud

    def store(self):
        return self.block_in_edge


class End(object):
    def __init__(self, args, replacer, columns, blocks):
        self.scheduler = get_model(method=replacer, args=args)
        self.columns = columns
        index, index_mat = indexMatConstruct(blocks, columns)
        self.indexer = index
        self.indexMat = index_mat
        self.block_in_end = None

    def locate(self, q):
        return fastDataLocate(q, self.columns, self.indexer, self.indexMat)

    def migrate(self, blocks, timestamp):
        self.scheduler.update(blocks, timestamp)
        self.block_in_end = self.scheduler.cache()

    def store(self):
        return self.block_in_end


class Container(object):
    def __init__(self, query, series):
        self.query = query
        self.series = series

    def get_query(self):
        return self.query

    def get_series(self, point=None):
        if point is None:
            return self.series
        else:
            return self.series[point]


def parse_arg():
    args = argparse.ArgumentParser()

    args.add_argument('--experiment', type=str, default='test')
    args.add_argument('--dataset', type=str, default='power')
    args.add_argument('--table_name', type=str, default='base')
    args.add_argument('--workload', type=str, default='standard')
    args.add_argument('--block_generate_strategy', type=str, default='pb_hbc',)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--split_point', type=int, default=6)
    args.add_argument('--warm_up', type=int, default=2)
    args.add_argument('--monitor_size', type=int, default=36)
    args.add_argument('--check_point', type=int, default=6)
    args.add_argument('--block_size', type=int, default=2048)
    args.add_argument('--page_size', type=int, default=512)
    args.add_argument('--page_generate_strategy', default='kd_tree')
    args.add_argument('--partition_generate_strategy', default='fk_means')
    args.add_argument('--page_order', default='hilbert_curve')
    args.add_argument('--filter_threshold', type=int, default=4)
    args.add_argument('--filter_ratio', type=float, default=0.6)
    args.add_argument('--num_cluster', type=int, default=16)
    args.add_argument('--post_partition', type=bool, default=True)
    args.add_argument('--cold_block_generator', default='kd_tree')
    args.add_argument('--cluster_method', default='fk_means')
    args.add_argument('--max_scan_blocks_num', type=int, default=1000)
    args.add_argument('--max_cluster_scale', type=float, default=0.08)
    args.add_argument('--K', type=int, default=8)
    args.add_argument('--kmeans_epochs', type=int, default=16)
    args.add_argument('--balance_ratio', type=float, default=0.0)
    args.add_argument('--curve_order', type=int, default=4)
    args.add_argument('--curve_filter_threshold', type=float, default=4)
    args.add_argument('--min_sequence_length', type=int, default=4)
    args.add_argument('--partition_strategy_with_curve', default='pdf')
    args.add_argument('--slope_threshold', type=float, default=1.0)
    args.add_argument('--partition_size', type=int, default=256)
    args.add_argument('--data_sample_ratio', type=float, default=0.01)
    args.add_argument('--cache_budget', type=int, default=81920)
    args.add_argument('--place_strategy', default='greedy')
    args.add_argument('--heat_model', default='current')
    args.add_argument('--forget_ratio', type=float, default=0.6)
    args.add_argument('--with_forecast', type=bool, default=False)
    args.add_argument('--future_weight', type=float, default=0.4)
    args.add_argument('--stat_win_size', type=int, default=12)
    args.add_argument('--slide_win_size', type=int, default=6)
    args.add_argument('--predict_horizon', type=int, default=3)
    args.add_argument('--hash_K', type=int, default=8)
    args.add_argument('--hash_L', type=int, default=16)
    args.add_argument('--lsh_threshold', type=float, default=0.8)
    args.add_argument('--max_vacancy_ratio', type=float, default=0.25)
    args.add_argument('--lsh_filter_threshold', type=float, default=0.2)
    args.add_argument('--cache_replacer', default='lru')
    args.add_argument('--num_page', type=int, default=256)
    args.add_argument('--warm_query_num', type=int, default=4)
    args.add_argument('--test_query_num', type=int, default=1024)
    args.add_argument('--evaluate_interval', type=int, default=128)

    args = args.parse_args()
    return args


def workload_split(zones, workload_batches, split_point):
    train, test = {}, {}
    for k in range(split_point):
        train[k] = []
        for z in zones:
            train[k] += workload_batches[k][z]
    for k in range(split_point, len(workload_batches)):
        test[k] = []
        for z in zones:
            test[k] += workload_batches[k][z]
    counter = 0
    workload = {}
    for k in train:
        workload[k] = {counter + off: train[k][off] for off in range(len(train[k]))}
        counter += len(train[k])
    for k in test:
        workload[k] = {counter + off: test[k][off] for off in range(len(test[k]))}
        counter += len(test[k])
    return workload


def workload_organize(zones, workload_batches, split_point, compress=False):
    train, test = [], []

    if compress:
        counter = {z: 0 for z in zones}
        for k in range(split_point):
            for z in zones:
                counter[z] += len(workload_batches[k][z])
        base = min(v for v in counter.values())
        counter = {z: math.ceil(counter[z] / base) for z in counter}
        for z in counter:
            holder = []
            for k in range(split_point):
                holder += workload_batches[k][z]
            sample = random.sample(holder, counter[z])
            train += list(sample)
    else:
        for k in range(split_point):
            holder = []
            for z in zones:
                holder += workload_batches[k][z]
            random.shuffle(holder)
            train += holder

    for k in range(split_point, len(workload_batches)):
        holder = []
        for z in zones:
            holder += workload_batches[k][z]
        random.shuffle(holder)
        test += holder
    return train, test


if __name__ == '__main__':
    args = parse_arg()
    random.seed(args.seed)

    data_path = DATA_ROOT / args.dataset
    table = load(data_path / f"{args.table_name}.pkl")
    columns = {c: table.columns[c].idx for c in table.columns.keys()}

    data = pd.read_csv(data_path / f'{args.table_name}.csv')
    data = pd.DataFrame(data)

    workload_path = DATA_ROOT / args.dataset / f'workload' / args.experiment
    workload = load(workload_path / f'{args.workload}.pkl')

    zone_path = DATA_ROOT / args.dataset / f'workload' / args.experiment
    zones = load(workload_path / f'{args.workload}_meta.pkl')

    blocks = BlockGenerator(args, data, None, columns).load('blocks')

    evaluator = Evaluator(columns, blocks)

    workload = workload_split(zones, workload, args.split_point)

    block_cache_hits = 0
    block_cache_miss = 0
    tuple_cache_hits = 0
    tuple_cache_miss = 0
    table_scan_size = 0
    query_counter = 0

    cloud = Cloud()
    edge = Edge(args, columns, blocks)
    end = End(args, args.cache_replacer, columns, blocks)

    i = args.split_point - args.warm_up
    while i < args.split_point + args.monitor_size:
        batch = workload[i]
        edge.monitor(batch)

        if i <= args.split_point:
            queries = [q for q in batch.values()]
            random.shuffle(queries)
            for q in queries:
                q_blocks = end.locate(q)
                end.migrate(q_blocks, i)
            block_in_cloud = edge.migrate()
            cloud.migrate(block_in_cloud)

        if i > args.split_point:
            queries = [q for q in batch.values()]
            random.shuffle(queries)
            for q in queries:
                query_counter += 1
                q_blocks = end.locate(q)
                block_in_end = end.store()

                scan_hits, scan_miss = evaluator.scan_size(q_blocks, block_in_end)
                table_scan_size += scan_hits + scan_miss
                cache_hits, cache_miss = evaluator.cache_hit(q, q_blocks, block_in_end, 'tuple')
                tuple_cache_hits += cache_hits
                tuple_cache_miss += cache_miss
                cache_hits, cache_miss = evaluator.cache_hit(q, q_blocks, block_in_end, 'block')
                block_cache_hits += cache_hits
                block_cache_miss += cache_miss

                end.migrate(q_blocks, i)
                if query_counter % args.evaluate_interval == 0:
                    print('Tuple Cache Hits Ratio {} || Block Cache Hits Ratio {} || Scan Ratio {}'.format(
                        round(tuple_cache_hits / (tuple_cache_hits + tuple_cache_miss), 5),
                        round(block_cache_hits / (block_cache_hits + block_cache_miss), 5),
                        round(table_scan_size / (len(data) * query_counter), 5)))

            block_in_cloud = edge.migrate()
            cloud.migrate(block_in_cloud)

        i += 1

    print('Tuple Cache Hits Ratio {} || Block Cache Hits Ratio {} || Scan Ratio {}'.format(
        round(tuple_cache_hits / (tuple_cache_hits + tuple_cache_miss), 5),
        round(block_cache_hits / (block_cache_hits + block_cache_miss), 5),
        round(table_scan_size / (len(data) * query_counter), 5)))

