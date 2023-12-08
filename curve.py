import numpy as np
import math
import time
from hilbertcurve.hilbertcurve import HilbertCurve


class SpaceFillingCurve(object):

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

    def __init__(self, args, pages, data, workload, columns):
        self.args = args
        self.data = data
        self.workload = workload
        self.columns = columns
        self.pages = pages
        self.pageIndex = {p: idx for idx, p in enumerate(self.pages)}
        self.indexPage = {v: k for k, v in self.pageIndex.items()}

    def page_rank(self):
        matrix = np.zeros(shape=(len(self.pages), len(self.columns)))
        for pid in self.pages:
            center = self.pages[pid].get_center()
            center = np.round(center * (10 ** self.args.curve_order))
            matrix[self.pageIndex[pid]] = center

        order = math.ceil(math.log(10 ** self.args.curve_order + 1, 2))
        space_filling_curve = HilbertCurve(p=order, n=len(self.columns))
        locations = space_filling_curve.distances_from_points(points=matrix)
        ranks = np.argsort(locations)
        sequence = [-1 for _ in range(len(self.pages))]
        for i, r in enumerate(ranks):
            sequence[r] = self.indexPage[i]
        return sequence

    def get_statis(self):
        statis = np.zeros(shape=(len(self.pages)))
        matrix = np.zeros(shape=(len(self.data), len(self.workload)))
        for idx, q in enumerate(self.workload):
            bitmap = q.generate_bitmap(sample=self.data.values, col2idx=self.columns)
            matrix[:, idx] = bitmap
        for pid in self.pages:
            encode = matrix[self.pages[pid].index]
            check_mat = np.sum(encode, axis=0)
            valid_q = np.where(check_mat > 0)[0]
            statis[self.pageIndex[pid]] = np.sum(encode) / (len(encode) * len(valid_q)) if np.sum(encode) > 0 else 0
        return statis

    def fast_get_statis(self):
        statis = np.zeros(shape=(len(self.pages)))
        from utility import fastDataLocate, indexMatConstruct
        mat = np.zeros(shape=(len(self.pages), len(self.workload)))
        mapper, index_mat = indexMatConstruct(blocks=self.pages, columns=self.columns, level='node')
        for i, q in enumerate(self.workload):
            routed_pages = fastDataLocate(q=q, columns=self.columns, mapper=mapper, index_mat=index_mat)
            routed_pages = [self.pageIndex[p] for p in routed_pages]
            mat[routed_pages, i] = 1
        freq_mat = np.sum(mat, axis=1)
        for i in range(len(freq_mat)):
            pid = mapper[i]
            statis[self.pageIndex[pid]] = freq_mat[i]
        return statis

    def get_cdf(self, pdf):
        cdf = []
        base = 0.0
        for s in pdf:
            base += s
            cdf.append(base)
        return cdf

    def appr_cdf(self, cdf, step):
        appr_cdf = [(cdf[i] + cdf[i + step]) / 2 for i in range(len(cdf) - step)]

        clusters = {}
        cls_idx = 0
        cursor = 0
        group = []
        gradiant = 0
        while cursor < len(appr_cdf):
            if len(group) == 0:
                gradiant = appr_cdf[cursor]
                group.append(cursor)
            else:
                if gradiant == 0 or appr_cdf[cursor] == 0:
                    slope_ratio = max(gradiant, appr_cdf[cursor])
                else:
                    slope_ratio = max(gradiant / appr_cdf[cursor], appr_cdf[cursor] / gradiant)
                if slope_ratio <= self.args.slope_threshold:
                    gradiant = (gradiant * len(group) + appr_cdf[cursor]) / (len(group) + 1)
                    group.append(cursor)
                else:
                    clusters[cls_idx] = group
                    gradiant = appr_cdf[cursor]
                    group = [cursor]
                    cls_idx += 1
            cursor += 1

        for k in clusters:
            clusters[k] = [i for i in range(clusters[k][0], clusters[k][-1] * step)]
        clusters[cls_idx] = [i for i in range(group[0], len(cdf))]

        return clusters

    def generate_cluster(self, sequence, statis, strategy):
        pdf = [statis[self.pageIndex[i]] for i in sequence]

        if strategy == 'pdf':
            clusters = {}
            cls_idx = 0

            if pdf[0] < self.args.curve_filter_threshold:
                cluster = self.Cluster(state='cold')
            else:
                cluster = self.Cluster(state='warm')
            cluster.add_item(item=0)
            clusters[cls_idx] = cluster
            cls_idx += 1

            for cursor in range(1, len(pdf)):
                if pdf[cursor] < self.args.curve_filter_threshold:
                    if clusters[cls_idx - 1].state == 'cold':
                        clusters[cls_idx - 1].add_item(item=cursor)
                    else:
                        cluster = self.Cluster(state='cold')
                        clusters[cls_idx] = cluster
                        clusters[cls_idx].add_item(item=cursor)
                        cls_idx += 1
                else:
                    if clusters[cls_idx - 1].state == 'cold':
                        cluster = self.Cluster(state='warm')
                        clusters[cls_idx] = cluster
                        clusters[cls_idx].add_item(item=cursor)
                        cls_idx += 1
                    else:
                        clusters[cls_idx - 1].add_item(item=cursor)

            cls_indexer = [k for k in clusters.keys()]
            cursor = 0
            shadow = -1
            while cursor < len(cls_indexer):
                cur = cls_indexer[cursor]
                if clusters[cur].state == 'warm':
                    shadow = cursor
                else:
                    if clusters[cur].get_size() > self.args.min_sequence_length:
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

            segments = {}
            for idx, cid in enumerate(clusters):
                segments[idx] = clusters[cid]
                segments[idx].items = [sequence[i] for i in clusters[cid].items]
            return segments

        elif strategy == 'cdf':
            raise NotImplementedError

        else:
            raise NotImplementedError

    def pipeline(self):
        since = time.time()
        sequence = self.page_rank()
        print('[Page Rank Along With Hilbert Curve] Time Consume : ', time.time() - since)
        since = time.time()
        statis = self.fast_get_statis()
        print('[Statis information Generation] Time Consume : ', time.time() - since)
        since = time.time()
        clusters = self.generate_cluster(sequence=sequence, statis=statis,
                                         strategy=self.args.partition_strategy_with_curve)
        print('[Hot Cold Pages Cluster] Time Consume : ', time.time() - since)
        return clusters

