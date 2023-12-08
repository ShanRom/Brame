import numpy as np
import pandas as pd
import time
import random
import argparse
from constants import OUTPUT_ROOT, DATA_ROOT
from structures import Block, CustomUnpickler
from utility import getLeaves, save, load
from table_partition import TablePartitioner
from data_reorganization import PartitionOrganizer


class BlockGenerator(object):
    def __init__(self, args, data, workload, columns):
        self.args = args
        self.data = data
        self.workload = workload
        self.columns = columns
        prompt = '{}+{}+{}'.format(args.dataset, args.workload, args.block_size)

        self.path = OUTPUT_ROOT / args.experiment / args.block_generate_strategy / prompt

    def createBlock(self, node):
        block = Block(block_id=node.block_id, data=self.data.loc[node.index])
        block.set_domains()
        return block

    def block_generate(self):

        blocks = {}
        Count = 0

        if self.args.block_generate_strategy == 'pb_hbc':

            since = time.time()
            partitions, cold_blocks = TablePartitioner(args=self.args, data=self.data,
                                                       workload=self.workload, columns=self.columns).pipeline()
            print('Table Partition Time Consume : {}'.format(time.time() - since))
            partition_scales = [len(part) for part in partitions.values()]
            print('Partition Scales : ', partition_scales)

            since = time.time()
            forest = PartitionOrganizer(args=self.args, data=self.data, workload=self.workload,
                                        columns=self.columns, partitions=partitions).pipeline(algorithm='pb_hbc')
            print('Data Reorganization Time Consume : {}'.format(time.time() - since))

            for b in cold_blocks:
                b.block_id = Count
                blocks[Count] = self.createBlock(node=b)
                Count += 1

            for k in forest:
                hot_blocks = forest[k]

                for b in hot_blocks.values():
                    bid = b.node_id
                    forest[k][bid].block_id = Count
                    blocks[Count] = self.createBlock(node=forest[k][bid])
                    Count += 1

            return blocks

        elif self.args.block_generate_strategy == 'pb_mbm':
            since = time.time()
            partitions, cold_blocks = TablePartitioner(args=self.args, data=self.data,
                                                       workload=self.workload, columns=self.columns).pipeline()
            print('Table Partition Time Consume : {}'.format(time.time() - since))
            partition_scales = [len(part) for part in partitions.values()]
            print('Partition Scales : ', partition_scales)

            since = time.time()
            forest = PartitionOrganizer(args=self.args, data=self.data, workload=self.workload,
                                        columns=self.columns, partitions=partitions).pipeline(algorithm='pb_mbm')
            print('Data Reorganization Time Consume : {}'.format(time.time() - since))

            for b in cold_blocks:
                b.block_id = Count
                blocks[Count] = self.createBlock(node=b)
                Count += 1

            for k in forest:
                hot_blocks = forest[k]

                for b in hot_blocks.values():
                    bid = b.node_id
                    forest[k][bid].block_id = Count
                    blocks[Count] = self.createBlock(node=forest[k][bid])
                    Count += 1

            return blocks

        elif self.args.block_generate_strategy == 'pb_gbm':
            since = time.time()
            partitions, cold_blocks = TablePartitioner(args=self.args, data=self.data,
                                                       workload=self.workload, columns=self.columns).pipeline()
            print('Table Partition Time Consume : {}'.format(time.time() - since))
            partition_scales = [len(part) for part in partitions.values()]
            print('Partition Scales : ', partition_scales)

            since = time.time()
            forest = PartitionOrganizer(args=self.args, data=self.data, workload=self.workload,
                                        columns=self.columns, partitions=partitions).pipeline(algorithm='pb_gbm')
            print('Data Reorganization Time Consume : {}'.format(time.time() - since))

            for b in cold_blocks:
                b.block_id = Count
                blocks[Count] = self.createBlock(node=b)
                Count += 1

            for k in forest:
                hot_blocks = forest[k]

                for b in hot_blocks.values():
                    bid = b.node_id
                    forest[k][bid].block_id = Count
                    blocks[Count] = self.createBlock(node=forest[k][bid])
                    Count += 1

            return blocks

        elif self.args.block_generate_strategy == 'qb_hbc':
            from qd_tree import QDTree

            since = time.time()
            agent = QDTree(args=self.args, data=self.data, columns=self.columns)
            agent.create_tree(workload=self.workload)
            partitions = agent.partition_generate()
            print('Table Partition Time Consume : {}'.format(time.time() - since))

            partition_scales = [len(part) for part in partitions.values()]
            print('Partition Scales : ', partition_scales)

            since = time.time()
            forest = PartitionOrganizer(args=self.args, data=self.data, workload=self.workload,
                                        columns=self.columns, partitions=partitions).pipeline(algorithm='qb_hbc')
            print('Data Reorganization Time Consume : {}'.format(time.time() - since))

            for k in forest:
                hot_blocks = forest[k]

                for b in hot_blocks.values():
                    bid = b.node_id
                    forest[k][bid].block_id = Count
                    blocks[Count] = self.createBlock(node=forest[k][bid])
                    Count += 1

            return blocks

        else:
            raise NotImplementedError

    def block_generate_with_indexer(self):
        from data_reorganization import HBC
        assert self.args.block_generate_strategy == 'pb_hbc'
        blocks = {}
        Count = 0
        since = time.time()
        hot_partitions, cold_forest, cold_indexer = TablePartitioner(args=self.args, data=self.data,
                                                                     workload=self.workload,
                                                                     columns=self.columns).pipeline(with_indexer=True)
        cold_forests = {}
        for i in range(len(cold_forest)):
            cold_forests[i] = (cold_forest[i], cold_indexer[i])
        print('Table Partition Time Consume : {}'.format(time.time() - since))

        since = time.time()
        hot_forests = {}
        for pid in hot_partitions:
            hot_forest, hot_indexer = HBC(args=self.args, data=self.data, workload=self.workload, columns=self.columns,
                                          index=hot_partitions[pid]).pipeline(with_indexer=True)
            hot_forests[pid] = (hot_forest, hot_indexer)

        print('Data Reorganization Time Consume : {}'.format(time.time() - since))

        hot_router = {}
        cold_router = {}

        Part = 0
        for k in cold_forests:
            leaves, meta = cold_forests[k]

            start = Count
            for b in leaves:
                b.block_id = Count
                blocks[Count] = self.createBlock(node=b)
                Count += 1

            cold_router[Part] = {
                'loc': (start, Count),
                'dom': meta
            }
            Part += 1

        Part = 0
        for k in hot_forests:
            leaves, meta = hot_forests[k]

            start = Count
            for b in leaves.values():
                b.block_id = Count
                blocks[Count] = self.createBlock(node=b)
                Count += 1

            hot_router[Part] = {
                'loc': (start, Count),
                'dom': meta
            }
            Part += 1

        return blocks, hot_router, cold_router

    def create(self):
        blocks = self.block_generate()

        save(obj=blocks, name='blocks', path=self.path)

    def load(self, name):

        if name == 'blocks':
            blocks = load(name=name, path=self.path)
            return blocks

        else:
            raise NotImplementedError


def workload_organize(zones, workload_batches, split_point):
    train, test = [], []
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


def parse_arg():
    args = argparse.ArgumentParser()

    args.add_argument('--experiment', type=str, default='test')
    args.add_argument('--dataset', type=str, default='power')
    args.add_argument('--table_name', type=str, default='base')
    args.add_argument('--workload', type=str, default='standard')
    args.add_argument('--block_generate_strategy', type=str, default='pb_hbc')
    args.add_argument('--split_point', type=int, default=8)
    args.add_argument('--warm_up', type=int, default=2)
    args.add_argument('--monitor_size', type=int, default=16)
    args.add_argument('--check_point', type=int, default=2)
    args.add_argument('--block_size', type=int, default=2048)
    args.add_argument('--page_size', type=int, default=512)
    args.add_argument('--page_generate_strategy', default='kd_tree')
    args.add_argument('--partition_generate_strategy', default='fk_means')
    args.add_argument('--page_order', default='hilbert_curve')
    args.add_argument('--filter_threshold', type=int, default=4)
    args.add_argument('--filter_ratio', type=float, default=0.6)
    args.add_argument('--num_cluster', type=int, default=16)
    args.add_argument('--cold_block_generator', default='kd_tree')
    args.add_argument('--cluster_method', default='fk_means')
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

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()

    data_path = DATA_ROOT / args.dataset
    table = CustomUnpickler(open(data_path / f"{args.table_name}.pkl", 'rb')).load()
    columns = {c: table.columns[c].idx for c in table.columns.keys()}

    data = pd.read_csv(data_path / f'{args.table_name}.csv')
    data = pd.DataFrame(data)

    workload_path = DATA_ROOT / args.dataset / f'workload' / args.experiment
    loader = open(workload_path / f'{args.workload}.pkl', 'rb')
    workload = CustomUnpickler(loader).load()
    loader.close()

    zone_path = DATA_ROOT / args.dataset / f'workload' / args.experiment
    loader = open(workload_path / f'{args.workload}_meta.pkl', 'rb')
    zones = CustomUnpickler(loader).load()
    loader.close()

    train, test = workload_organize(workload_batches=workload, split_point=args.split_point, zones=zones)
    agent = BlockGenerator(args=args, data=data, workload=train, columns=columns)
    agent.create()
