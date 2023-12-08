import random
import time
import argparse
import pandas as pd
import numpy as np
from typing import List
import copy

from structures import Table, Query
from craft_series import series_generate
from constants import DATA_ROOT
from structures import CustomUnpickler, save


class WorkloadGenerator(object):
    def __init__(self, args, dataset, table):
        self.args = args
        self.data = dataset
        self.sample = self.data.sample(frac=args.sample_frac)
        self.table = table
        self.cols2idx = table.col2idx
        self.idx2cols = table.idx2col
        self.nrows = table.nrows
        self.ncols = table.ncols
        self.columns = table.columns

    def template_generate(self, pools=None):
        templates = []
        if pools is None:
            pools = [k for k in self.cols2idx.keys()]
        while len(templates) < self.args.n_templates:
            pick_columns = random.sample(pools,
                                         random.randint(a=self.args.col_n_min, b=self.args.col_n_max))
            templates.append(pick_columns)
        return templates

    def query_zone_select(self, template, strategy, scale):

        if strategy == 'random':
            center = np.random.random(size=self.ncols)

        elif strategy == 'standard':
            data = self.data.values
            t_idx = np.random.randint(0, self.nrows, size=1)
            center = data[t_idx][0]

        else:
            raise NotImplementedError

        width = np.zeros(self.ncols)
        for c in template:
            c_idx = self.cols2idx[c]
            width[c_idx] = scale * random.random()

        left = center - width / 2
        right = center + width / 2
        left[left < 0] = 0.0
        right[right > 1] = 1.0

        predicates = {c: (left[self.cols2idx[c]], right[self.cols2idx[c]]) for c in template}
        triplets = []

        for c in predicates:
            l, r = predicates[c]

            if l == 0.0 and r == 1.0:
                return False, []

            elif l == 0.0:
                op = '<='
                val = r
                triplets.append((c, op, val))

            elif r == 1.0:
                op = '>='
                val = l
                triplets.append((c, op, val))

            else:
                if np.random.random() > self.args.oneside_prob or len(template) == 1:
                    op = '[]'
                    triplets.append((c, op, (l, r)))

                else:
                    if np.random.random() < 0.5:
                        op = '<='
                        val = r
                        triplets.append((c, op, val))

                    else:
                        op = '>='
                        val = l
                        triplets.append((c, op, val))

        return True, triplets

    def query_generate(self, template, strategy, scale):

        if self.args.use_sample:
            data = self.sample.values
        else:
            data = self.data.values

        valid = False
        triplets = None
        factor = scale
        fail_count = 0

        while not valid:

            valid, triplets = self.query_zone_select(template, strategy, factor)

            fail_count += 1

        nrows = len(data)
        bitmap = np.ones(nrows).astype(bool)
        predicates = {}

        since = time.time()
        for triplet in triplets:
            col, op, val = triplet
            p = self.cols2idx[col]
            if op == '[]':
                bitmap &= (data[:, p] >= val[0])
                bitmap &= (data[:, p] <= val[1])
            elif '<' in op:
                bitmap &= (data[:, p] <= val)
            elif '>' in op:
                bitmap &= (data[:, p] >= val)
            else:
                continue
            predicates[col] = (op, val)

        if self.args.use_sample:
            cardinality = int(np.sum(bitmap, axis=0) / self.args.sample_frac)
        else:
            cardinality = int(np.sum(bitmap, axis=0))

        latency = time.time() - since

        query = Query(
            predicates=predicates,
            card=cardinality,
            cost=latency * 1e3
        )

        return query, bitmap

    def query_extend(self, query, scale_factor):
        predicates = query.predicates
        if len(predicates.keys()) == 1:
            n_extend_dim = np.ones(scale_factor, dtype=int)
        else:
            n_extend_dim = np.random.randint(low=1, high=len(predicates), size=scale_factor)

        columns = [c for c in predicates.keys()]

        query_batches = []

        for idx in range(scale_factor):

            extend_predicates = predicates

            extend_dim = np.random.choice(a=columns, size=n_extend_dim[idx], replace=False)
            ratio = self.args.vary_ratio
            factor = np.random.uniform(low=-ratio, high=ratio, size=n_extend_dim[idx])

            for i in range(n_extend_dim[idx]):
                dim = extend_dim[i]
                op, val = predicates[dim]
                if op == '[]':
                    l, r = val
                    w = (r - l) / 2
                    c = (l + r) / 2
                    extend_w = w + w * factor[i]
                    extend_l, extend_r = c - extend_w, c + extend_w
                    if extend_l < 0.0 and extend_r > 1.0:
                        if factor[i] > 0:
                            extend_w = w - w * factor[i]
                            extend_l, extend_r = c - extend_w, c + extend_w
                        else:
                            assert False
                    extend_l = max(0.0, extend_l)
                    extend_r = min(1.0, extend_r)

                    if np.random.random() > self.args.oneside_prob:
                        extend_val = (extend_l, extend_r)
                    else:
                        if np.random.random() < 0.5:
                            extend_val = (l, extend_r)
                        else:
                            extend_val = (extend_l, r)

                elif '<' in op:
                    extend_val = val + val * factor[i]
                    if extend_val >= 1.0:
                        if factor[i] > 0:
                            extend_val = val - val * factor[i]
                        else:
                            assert False

                elif '>' in op:
                    extend_val = val + val * factor[i]
                    if extend_val <= 0.0:
                        if factor[i] < 0:
                            extend_val = val - val * factor[i]
                        else:
                            assert False

                else:
                    raise NotImplementedError

                extend_predicates[dim] = (op, extend_val)

            extend_query = Query(predicates=extend_predicates)
            query_batches.append(extend_query)

        return query_batches

    def query_complete(self, query_batch):

        if self.args.use_sample:
            data = self.sample.values
        else:
            data = self.data.values

        nrows = len(data)
        processed_q_batch = []

        for q in query_batch:

            inds = np.ones(nrows).astype(bool)
            predicates = {}

            since = time.time()
            for col in q.predicates.keys():
                p = self.cols2idx[col]
                op, val = q.predicates[col]
                if op == '[]':
                    inds &= (data[:, p] >= val[0])
                    inds &= (data[:, p] <= val[1])
                elif '<' in op:
                    inds &= (data[:, p] <= val)
                elif '>' in op:
                    inds &= (data[:, p] >= val)
                else:
                    continue

                predicates[col] = (op, val)
            latency = time.time() - since

            if self.args.use_sample:
                cardinality = int(np.sum(inds, axis=0) / self.args.sample_frac)
            else:
                cardinality = int(np.sum(inds, axis=0))

            complete_q = Query(
                predicates=predicates,
                card=cardinality,
                cost=latency
            )
            processed_q_batch.append(complete_q)

        return processed_q_batch

    def represent_workload_generate(self, template, size):

        workload = []
        bitmaps = []
        nrows = len(self.data) if not self.args.use_sample else len(self.sample)

        scale = 0.4

        init = True
        while init:
            query, bitmap = self.query_generate(template, strategy='standard', scale=scale)
            legal = query.check_valid()
            if not legal:
                continue
            sel = np.sum(bitmap) / nrows
            if self.args.sel_lb < sel < self.args.sel_ub:
                workload.append(query)
                bitmaps.append(bitmap)
                init = False

        sels = []

        while len(workload) < size:
            query, bitmap = self.query_generate(template, strategy='standard', scale=scale)
            legal = query.check_valid()
            if not legal:
                continue

            sel = np.sum(bitmap) / nrows
            sels.append(sel)
            if len(sels) > 0 and len(sels) % 100 == 0:
                print(np.mean(sels))
            if not(self.args.sel_lb <= sel <= self.args.sel_ub):
                if sel < self.args.sel_lb:
                    scale /= np.random.random()
                    scale = max(0.01, scale)
                elif sel > self.args.sel_ub:
                    scale *= np.random.random()
                    scale = min(2.0, scale)
            else:
                valid = True
                for item in bitmaps:
                    intersection = item & bitmap
                    union = item | bitmap
                    sim = np.sum(intersection) / np.sum(union)
                    if sim > self.args.sim_threshold:
                        valid = False
                if valid:
                    workload.append(query)
                    bitmaps.append(bitmap)

        return workload

    def represent_workload_generate_wrapper(self, templates):
        workloads = []
        size = self.args.n_represent_q / self.args.n_templates
        for t in templates:
            workload = self.represent_workload_generate(t, size)
            workloads += workload
        return workloads

    def query_extend_and_generate(self, query, scale_factor):
        predicates = copy.deepcopy(query.predicates)

        if self.args.use_sample:
            data = self.sample.values
        else:
            data = self.data.values

        nrows = len(data)

        if len(predicates.keys()) == 1:
            n_extend_dim = np.ones(scale_factor, dtype=int)
        else:
            n_extend_dim = np.random.randint(low=1, high=len(predicates), size=scale_factor)

        columns = [c for c in predicates.keys()]

        query_batches = []

        count = 0
        ratio = self.args.vary_ratio
        fail_count = 0
        try_count = 0

        while count < scale_factor:

            extend_predicates = copy.deepcopy(predicates)

            extend_dim = np.random.choice(a=columns, size=n_extend_dim[count], replace=False)
            factor = np.random.uniform(low=-ratio, high=ratio, size=n_extend_dim[count])

            for i in range(n_extend_dim[count]):
                dim = extend_dim[i]
                op, val = predicates[dim]
                if op == '[]':
                    l, r = val
                    w = (r - l) / 2
                    c = (l + r) / 2
                    extend_w = w + w * factor[i]
                    extend_l, extend_r = c - extend_w, c + extend_w
                    if extend_l < 0.0 and extend_r > 1.0:
                        if factor[i] > 0:
                            extend_w = w - w * factor[i]
                            extend_l, extend_r = c - extend_w, c + extend_w
                        else:
                            assert False
                    extend_l = max(0.0, extend_l)
                    extend_r = min(1.0, extend_r)

                    if np.random.random() > self.args.oneside_prob:
                        extend_val = (extend_l, extend_r)
                    else:
                        if np.random.random() < 0.5:
                            extend_val = (l, extend_r)
                        else:
                            extend_val = (extend_l, r)

                elif '<' in op:
                    extend_val = val + val * factor[i]
                    if extend_val >= 1.0:
                        if factor[i] > 0:
                            extend_val = val - val * factor[i]
                        else:
                            assert False

                elif '>' in op:
                    extend_val = val + val * factor[i]
                    if extend_val <= 0.0:
                        if factor[i] < 0:
                            extend_val = val - val * factor[i]
                        else:
                            assert False

                else:
                    raise NotImplementedError

                extend_predicates[dim] = (op, extend_val)

            inds = np.ones(nrows).astype(bool)

            since = time.time()
            for col in extend_predicates.keys():
                p = self.cols2idx[col]
                op, val = extend_predicates[col]
                if op == '[]':
                    inds &= (data[:, p] >= val[0])
                    inds &= (data[:, p] <= val[1])
                elif '<' in op:
                    inds &= (data[:, p] <= val)
                elif '>' in op:
                    inds &= (data[:, p] >= val)
                else:
                    continue

            latency = time.time() - since

            if self.args.use_sample:
                cardinality = int(np.sum(inds, axis=0) / self.args.sample_frac)
            else:
                cardinality = int(np.sum(inds, axis=0))

            sel = np.sum(inds, axis=0) / nrows

            if self.args.sel_lb <= sel <= self.args.sel_ub:

                complete_q = Query(
                    predicates=extend_predicates,
                    card=cardinality,
                    cost=latency
                )

                query_batches.append(complete_q)
                count += 1
                ratio = self.args.vary_ratio
                fail_count = 0
                try_count = 0

            else:

                try_count += 1

                if sel > self.args.sel_ub * 2:
                    fail_count += 1
                if sel < self.args.sel_lb / 2:
                    fail_count -= 1

                if fail_count == 10 or fail_count == -10:
                    n_extend_dim[count] = 1

                if fail_count % 20 == 0 and fail_count != 0:
                    ratio = ratio * 2
                    ratio = min(ratio, 0.001)
                    ratio = max(ratio, 2)

                if try_count > 50:
                    if self.args.sel_lb / 4 <= sel <= self.args.sel_ub * 4:
                        complete_q = Query(
                            predicates=extend_predicates,
                            card=cardinality,
                            cost=latency
                        )

                        query_batches.append(complete_q)
                        count += 1
                        ratio = self.args.vary_ratio
                        fail_count = 0
                        try_count = 0

                if try_count == 100:
                    query.sql_generate(table_name=self.args.dataset)
                    print(query.sql, query.card)

        return query_batches


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
    args.add_argument('--use_sample', type=bool, default=True)
    args.add_argument('--sample_frac', type=float, default=0.1)

    args.add_argument('--is_strict', type=bool, default=True)
    args.add_argument('--pattern', choices=['same', 'similar'], default='same')

    args.add_argument('--period', type=int, default=64)

    args.add_argument('--n_represent_q', type=int, default=32)
    args.add_argument('--n_templates', type=int, default=8)

    args.add_argument('--col_n_min', type=int, default=1)
    args.add_argument('--col_n_max', type=int, default=4)

    args.add_argument('--oneside_prob', type=float, default=0.25)
    args.add_argument('--vary_ratio', type=float, default=0.2)

    args.add_argument('--sel_ub', type=float, default=0.04)
    args.add_argument('--sel_lb', type=float, default=0.01)
    args.add_argument('--sim_threshold', type=float, default=0.2)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()

    data_path = DATA_ROOT / args.dataset

    data = pd.read_csv(data_path / f"{args.table_name}.csv")
    data = pd.DataFrame(data)

    table = CustomUnpickler(open(data_path / f"{args.table_name}.pkl", 'rb')).load()

    workload_generator = WorkloadGenerator(args, data, table)

    since = time.time()

    templates = workload_generator.template_generate()
    represent_workload = workload_generator.represent_workload_generate_wrapper(templates)

    Containers = []

    for query in represent_workload:
        series = series_generate(args.period, with_offset=True, with_bias=True)
        query.sql_generate(args.dataset)
        entity = Container(
            query=query,
            series=series
        )
        Containers.append(entity)

    Containers = {idx: item for idx, item in enumerate(Containers)}

    since = time.time()

    workload_batches = {}
    for timestamp in range(args.period):
        batch = {}
        for idx, entity in Containers.items():
            query = entity.get_query()
            freq = entity.get_series(point=timestamp)
            if freq == 0:
                continue
            else:
                if args.pattern == 'similar':
                    if args.is_strict:
                        q_batch = workload_generator.query_extend_and_generate(query, freq)
                    else:
                        q_batch = workload_generator.query_extend(query, freq)
                        q_batch = workload_generator.query_complete(q_batch)
                    batch[idx] = q_batch
                elif args.pattern == 'same':
                    batch[idx] = [query for _ in range(freq)]
                else:
                    raise NotImplementedError
        workload_batches[timestamp] = batch

        if timestamp % 1 == 0 and timestamp > 0:
            end = time.time() - since
            print('Finishing Workload Generation For {} Points ! [{} seconds]'.format(timestamp, end))
            since = time.time()

    save_path = DATA_ROOT / args.dataset / f'workload' / args.experiment
    save_path.mkdir(parents=True, exist_ok=True)
    save(workload_batches, file=save_path / f'{args.workload}.pkl')
    save(Containers, file=save_path / f'{args.workload}_meta.pkl')
