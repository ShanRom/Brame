import numpy as np
import time
import copy


class QDTree(object):
    class Node(object):
        def __init__(self, node_id, data, workload, parent_id):
            self.node_id = node_id

            self.data = data
            self.size = len(self.data)
            self.workload = workload

            self.domains = None
            self.predicates = []

            self.is_leaf = True
            self.can_split = True

            self.left_child = None
            self.right_child = None
            self.parent = parent_id

        def set_domains(self, domains):
            self.domains = domains

    def __init__(self, args, data, columns):
        self.args = args
        self.data = data
        self.columns = columns
        self.partition_tree = None
        self.nodeCounter = 0

    def get_leaves(self, for_split=True):
        leaves = []
        if for_split:
            for nid in self.partition_tree:
                if self.partition_tree[nid].is_leaf and self.partition_tree[nid].can_split:
                    leaves.append(self.partition_tree[nid])
        else:
            for nid in self.partition_tree:
                if self.partition_tree[nid].is_leaf:
                    leaves.append(self.partition_tree[nid])

        return leaves

    def candidate_generate(self, leaf):
        candidate_cuts = []
        for query in leaf.workload:
            cols, ops, vals = query.predicate_extract()
            for i in range(len(cols)):
                if ops[i] == '>=':
                    ops[i] = '<'
                if ops[i] == '>':
                    ops[i] = '<='
                candidate_cuts.append((cols[i], ops[i], vals[i]))
        return candidate_cuts

    def data_route(self, data, split_col, split_op, split_val):
        if '=' in split_op:
            data2left = data[data[split_col] <= split_val]
            data2right = data[data[split_col] > split_val]
        else:
            data2left = data[data[split_col] < split_val]
            data2right = data[data[split_col] >= split_val]
        return data2left, data2right

    def query_route(self, workload, split_col, split_op, split_val):
        query2left = []
        query2right = []
        for query in workload:
            low, up = query.domain_extract(col_name=split_col)
            if low <= split_val <= up:
                query2left.append(query)
                query2right.append(query)
            elif split_val < low:
                query2right.append(query)
            elif split_val > up:
                query2left.append(query)
            else:
                raise NotImplementedError

        return query2left, query2right

    def gain_calculate(self, leaf, split_col, split_op, split_val, partition_size):
        query2left, query2right = self.query_route(leaf.workload, split_col, split_op, split_val)
        data2left, data2right = self.data_route(leaf.data, split_col, split_op, split_val)
        if len(data2left) < partition_size or len(data2right) < partition_size:
            return False, -1, len(data2left)
        else:
            scan_cost = len(query2left) * len(data2left) + len(query2right) * len(data2right)
            gain = len(leaf.workload) * len(leaf.data) - scan_cost
            if gain < 0:
                return False, gain, len(data2left)
            return True, gain, len(data2left)

    def apply_split(self, leaf, split_col, split_op, split_val, partition_size):
        assert split_op == '<' or split_op == '<='

        predicate = (split_col, split_op, split_val)
        filters = leaf.predicates

        left_workload, right_workload = self.query_route(workload=leaf.workload, split_col=split_col,
                                                         split_op=split_op, split_val=split_val)
        left_data, right_data = self.data_route(leaf.data, split_col, split_op, split_val)

        left_domains = copy.deepcopy(leaf.domains)
        left_domains[self.columns[split_col]][1] = split_val
        left_filters = copy.deepcopy(filters)
        left_filters.append(predicate)

        right_domains = copy.deepcopy(leaf.domains)
        right_domains[self.columns[split_col]][0] = split_val
        right_filters = copy.deepcopy(filters)
        reverse_split_op = split_op.replace('<', '>')
        right_filters.append((split_col, reverse_split_op, split_val))

        self.partition_tree[leaf.node_id].predicates.append(predicate)
        self.partition_tree[leaf.node_id].is_leaf = False
        self.partition_tree[leaf.node_id].can_split = False
        del self.partition_tree[leaf.node_id].workload
        del self.partition_tree[leaf.node_id].data
        self.partition_tree[leaf.node_id].left_child = self.nodeCounter + 1
        self.partition_tree[leaf.node_id].right_child = self.nodeCounter + 2

        child_left = self.Node(node_id=self.nodeCounter + 1,
                               data=left_data,
                               workload=left_workload,
                               parent_id=leaf.node_id)
        child_left.set_domains(domains=left_domains)
        child_left.predicates = left_filters
        child_right = self.Node(node_id=self.nodeCounter + 2,
                                data=right_data,
                                workload=right_workload,
                                parent_id=leaf.node_id)
        child_right.set_domains(domains=right_domains)
        child_right.predicates = right_filters

        if child_left.size < 2 * partition_size:
            child_left.can_split = False

        if child_right.size < 2 * partition_size:
            child_right.can_split = False

        self.partition_tree[self.nodeCounter + 1] = child_left
        self.partition_tree[self.nodeCounter + 2] = child_right
        self.nodeCounter += 2

    def try_split_node(self, leaf, candidate_cuts, partition_size):
        CanSplit = False

        max_gain, max_gain_split_col, max_gain_split_op, max_gain_split_val = -1, 0, '', 0

        for triple in candidate_cuts:
            split_col, split_op, split_val = triple
            valid, gain, _ = self.gain_calculate(leaf=leaf, split_col=split_col, split_op=split_op, split_val=split_val,
                                                 partition_size=partition_size)
            if valid and gain > max_gain:
                max_gain = gain
                max_gain_split_col = split_col
                max_gain_split_op = split_op
                max_gain_split_val = split_val

        if max_gain > 0:
            self.apply_split(leaf=leaf, split_col=max_gain_split_col, split_op=max_gain_split_op,
                             split_val=max_gain_split_val, partition_size=partition_size)
            CanSplit = True
        else:
            self.partition_tree[leaf.node_id].can_split = False

        return CanSplit

    def create_tree(self, workload):

        data = self.data.sample(frac=self.args.data_sample_ratio)

        construct_time = 0

        root = self.Node(data=data, workload=workload, node_id=self.nodeCounter, parent_id=-1)
        root.set_domains(domains=np.concatenate((np.zeros(len(self.columns)).reshape(-1, 1),
                                                 np.ones(len(self.columns)).reshape(-1, 1)), axis=1))

        self.partition_tree = {root.node_id: root}

        CanSplit = True
        layer = 0
        while CanSplit:
            layer += 1
            since = time.time()
            CanSplit = False
            leaves = self.get_leaves(for_split=True)
            if len(leaves) == 0:
                break
            for leaf in leaves:

                if leaf.size < 2 * self.args.partition_size:
                    leaf.partition_able = False
                    continue

                candidate_cuts = self.candidate_generate(leaf)

                scan_able = self.try_split_node(leaf, candidate_cuts, partition_size=self.args.partition_size)
                CanSplit = True if scan_able else CanSplit
            epoch_time = time.time() - since
            construct_time += epoch_time

    def get_index(self, domains):
        data = self.data.values
        ind = np.ones(len(self.data)).astype(bool)
        for cid in range(len(domains)):
            low, up = domains[cid][0], domains[cid][1]
            if low > 0.0:
                ind &= (data[:, cid] >= low)
            if up < 1.0:
                ind &= (data[:, cid] <= up)
        index = np.where(ind > 0)[0]
        return index

    def partition_generate(self):
        assert self.partition_tree is not None
        partCounter = 0
        partitions = {}
        leaves = self.get_leaves(for_split=False)
        for leaf in leaves:
            index = self.get_index(domains=leaf.domains)
            partitions[partCounter] = index
            partCounter += 1
        return partitions
