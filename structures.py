from typing import Dict, NamedTuple, Optional, Tuple, List, Any
import numpy as np
import pandas as pd
import pickle


from constants import PKL_PROTO


class Column(NamedTuple):
    name: str
    idx: int
    min_val: Any
    max_val: Any
    n_distinct: int
    vocab: List[Any] or Any


class Table(NamedTuple):
    name: str
    nrows: int
    ncols: int
    columns: Dict[str, Column]
    col2idx: Dict[str, int]
    idx2col: Dict[int, str]


class Query(object):
    def __init__(self, predicates: Dict[str, Tuple[str, Any]], card=None, cost=None, bitmap=None):
        self.sql = None
        self.predicates = predicates
        self.card = card
        self.cost = cost
        self.bitmap = bitmap

    def sql_generate(self, table_name):
        triplets = []
        if self.predicates is {}:
            SQL = 'SELECT * FROM {}'.format(table_name)
        else:
            for col, prd in self.predicates.items():
                op, val = prd
                if op in ['<', '<=', '>', '>=']:
                    triple = '{} {} {}'.format(col, op, val)
                    triplets.append(triple)
                else:
                    if '(' in op:
                        triple = '{} {} {}'.format(col, '>', val[0])
                        triplets.append(triple)
                    else:
                        triple = '{} {} {}'.format(col, '>=', val[0])
                        triplets.append(triple)
                    if ')' in op:
                        triple = '{} {} {}'.format(col, '<', val[1])
                        triplets.append(triple)
                    else:
                        triple = '{} {} {}'.format(col, '<=', val[1])
                        triplets.append(triple)
            prd_clause = ' AND '.join(triplets)
            SQL = 'SELECT * FROM {} WHERE {}'.format(table_name, prd_clause)
        self.sql = SQL

    def set_card(self, card):
        self.card = card

    def set_cost(self, cost):
        self.cost = cost

    def get_sql(self):
        return self.sql

    def get_card(self):
        return self.card

    def get_cost(self):
        return self.cost

    def column_extract(self):
        columns = [c for c in self.predicates.keys()]
        return sorted(columns)

    def domain_extract(self, col_name, pattern='win', vec_len=None):
        if pattern == 'win':
            if col_name not in self.predicates.keys():
                return 0.0, 1.0
            else:
                op = self.predicates[col_name][0]
                val = self.predicates[col_name][1]
                if op == '>' or op == '>=':
                    return val, 1.0
                elif op == '<' or op == '<=':
                    return 0.0, val
                elif op in ['[]', '(]', '[)', '()']:
                    return val[0], val[1]
                else:
                    raise NotImplementedError

        elif pattern == 'vec':
            assert vec_len is not None
            if col_name not in self.predicates.keys():
                return np.ones(vec_len)
            else:
                vec = np.zeros(vec_len)
                op = self.predicates[col_name][0]
                val = self.predicates[col_name][1]
                if op == '>' or '>=':
                    vec[int(vec_len * val):] = 1
                    return vec
                elif op == '<' or '<=':
                    vec[:int(vec_len * val + 1)] = 1
                    return vec
                elif op in ['[]', '(]', '[)', '()']:
                    vec[int(vec_len * val[0]): int(vec_len * val[1])] = 1
                    return vec
                else:
                    raise NotImplementedError

    def domains_extract(self, columns=None, pattern='win', vec_len=None):
        if columns is None:
            columns = self.column_extract()
        domains = {c: self.domain_extract(c, pattern=pattern, vec_len=vec_len) for c in columns}
        return domains

    def np_domains(self, col2idx):
        domains = self.domains_extract()
        box = np.zeros(shape=(len(col2idx), 2))
        for c in col2idx:
            box[col2idx[c]][0], box[col2idx[c]][1] = domains[c][0], domains[c][1]
        return box

    def predicate_extract(self, col_name=None):

        cols, ops, vals = [], [], []

        if col_name is not None:
            if col_name not in self.predicates.keys():
                return None, None, None
            col = col_name
            op, val = self.predicates[col_name]
            if op in ['<', '<=', '>', '>=']:
                cols.append(col)
                ops.append(op)
                vals.append(val)
            else:
                if '(' in op:
                    cols.append(col)
                    ops.append('>')
                    vals.append(val[0])
                else:
                    cols.append(col)
                    ops.append('>=')
                    vals.append(val[0])
                if ')' in op:
                    cols.append(col)
                    ops.append('<')
                    vals.append(val[1])
                else:
                    cols.append(col)
                    ops.append('<=')
                    vals.append(val[1])

        else:
            columns = self.column_extract()
            for col in columns:
                op, val = self.predicates[col]
                if op in ['<', '<=', '>', '>=']:
                    cols.append(col)
                    ops.append(op)
                    vals.append(val)
                else:
                    if '(' in op:
                        cols.append(col)
                        ops.append('>')
                        vals.append(val[0])
                    else:
                        cols.append(col)
                        ops.append('>=')
                        vals.append(val[0])
                    if ')' in op:
                        cols.append(col)
                        ops.append('<')
                        vals.append(val[1])
                    else:
                        cols.append(col)
                        ops.append('<=')
                        vals.append(val[1])

        return cols, ops, vals

    def generate_bitmap(self, sample, col2idx):
        domains = self.domains_extract()
        ind = np.ones(len(sample)).astype(bool)
        for c in domains.keys():
            c_id = col2idx[c]
            low, up = domains[c][0], domains[c][1]
            if low > 0.0:
                ind &= (sample[:, c_id] >= low)
            if up < 1.0:
                ind &= (sample[:, c_id] <= up)
        self.bitmap = ind
        return self.bitmap

    def check_valid(self):
        domains = self.domains_extract()
        for c in domains:
            if domains[c][0] < 0 or domains[c][1] > 1:
                return False
        return True


class Block(object):
    def __init__(self, block_id, data=None):
        self.block_id = block_id

        self.domains = None
        self.data = data
        self.size = len(self.data)

    def set_data(self, data):
        self.data = data
        self.size = len(self.data)

    def data_route(self, data, replace=False):
        assert self.domains is not None
        columns = self.data.columns
        if type(data) is pd.DataFrame:
            data = data.values
        check_low = np.amin(data - self.domains[:, 0].reshape(1, -1), axis=1)
        check_up = np.amin(self.domains[:, 1].reshape(1, -1) - data, axis=1)
        ind_satisfy_low = np.where(check_low >= 0.0)
        ind_satisfy_up = np.where(check_up >= 0.0)
        ind = np.intersect1d(ind_satisfy_low, ind_satisfy_up)
        satisfied_data = data[ind, :]
        satisfied_data = pd.DataFrame(satisfied_data, columns=columns)
        if replace:
            self.data = satisfied_data
        else:
            self.data = self.data.append(satisfied_data, ignore_index=True)
        self.size = len(self.data)

    def set_domains(self):
        assert self.data is not None
        min_vals = self.data.min().values.reshape(-1, 1)
        max_vals = self.data.max().values.reshape(-1, 1)
        self.domains = np.concatenate((min_vals, max_vals), axis=1)
        assert self.domains.shape[1] == 2


class Template(object):
    def __init__(self, template_id, q, blocks):
        self.template_id = template_id
        self.blocks = None
        self.workload = [q]
        self.hits = 1
        self.series = []
        self.sketch = {b: 1 for b in blocks}

    def add_hits(self):
        self.hits += 1

    def update_series(self):
        self.series.append(self.hits)
        self.hits = 0

    def update(self, q, blocks):
        self.workload.append(q)
        for b in blocks:
            if b not in self.sketch:
                self.sketch[b] = 1
            else:
                self.sketch[b] += 1

    def get_blocks(self, threshold):
        blocks = [b for b in self.sketch if self.sketch[b] >= int(threshold * len(self.workload))]
        self.blocks = blocks
        return blocks

    def get_weighted_blocks(self):
        blocks = {b: self.sketch[b] / len(self.workload) for b in self.sketch}
        return blocks

    def get_series(self, win_size):
        if len(self.series) <= win_size:
            return self.series
        else:
            return self.series[-win_size:]

    def predict(self, win_size, horizon):
        win_size = min(win_size, len(self.series))
        series = self.get_series(win_size)

        length = len(series)
        while len(series) < length + horizon:
            point = sum(series[-horizon:])
            series.append(int(point / horizon))
        return series[-horizon:]


OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal
}


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


def save(obj, file):
    writer = open(file, 'wb')
    pickle.dump(obj, writer, protocol=PKL_PROTO)


def load(file):
    loader = open(file, 'rb')
    obj = pickle.load(loader)
    return obj
