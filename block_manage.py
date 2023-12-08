import numpy as np
import copy


from utility import knapsackSolver, indexMatConstruct, fastDataLocate


class Listener(object):
    def __init__(self, block, stat_win_size, slide_win_size, horizon):
        self.block_id = block.block_id
        self.size = len(block.data)
        self.domains = block.domains

        self.temperature = 0.0
        self.hits = 0.0

        self.future_heat = 0.0
        self.future_hits = []

        self.stat_win_size = stat_win_size
        self.slide_win_size = slide_win_size
        self.horizon = horizon
        self.windows = []

    def compute_temperature(self, forget_ratio, with_forecast=False):

        self.temperature = self.temperature * (1 - forget_ratio) + self.hits * forget_ratio
        self.hits = 0

        if with_forecast:
            assert len(self.future_hits) > 0

            heat_pool = []
            heat = copy.deepcopy(self.temperature)
            for p in self.future_hits:
                heat = heat * (1 - forget_ratio) + p * forget_ratio
                heat_pool.append(heat)

            future_heat = 0
            w = 1.0
            base = 0.0
            for h in heat_pool:
                base += w
                future_heat += h * w
                w = w * forget_ratio
            self.future_heat = future_heat / base
            self.future_hits = []

    def update(self, with_forecast=False):

        if self.stat_win_size > len(self.windows):
            self.windows.append(self.hits)
        else:
            self.windows.pop(0)
            self.windows.append(self.hits)

        if with_forecast:
            win_size = min(len(self.windows), self.slide_win_size)
            series = self.windows[-win_size:]

            length = len(series)
            while len(series) < length + self.horizon:
                point = sum(series[-self.horizon:])
                series.append(int(point / self.horizon))
            self.future_hits = series[-self.horizon:]


class BlockScheduler(object):
    def __init__(self, args, blocks, columns):
        self.args = args
        self.columns = columns

        mapper, index_mat = indexMatConstruct(blocks=blocks, columns=columns, level='block')

        self.blockIndexer = mapper
        self.blockIndexMat = index_mat

        segments = {}
        for bid in blocks.keys():
            listener = Listener(blocks[bid], stat_win_size=args.stat_win_size, slide_win_size=args.slide_win_size,
                                horizon=args.predict_horizon)
            segments[bid] = listener
        self.blocks = segments
        self.partitions = blocks

        self.block_in_cache = []
        self.block_in_cloud = []

        self.map = {k: [] for k in self.blocks.keys()}

    def query_route(self, q):
        return fastDataLocate(q=q, columns=self.columns, mapper=self.blockIndexer, index_mat=self.blockIndexMat)

    def update(self, batch, with_forecast=False):
        for q in batch.values():
            blocks = self.query_route(q=q)
            for bid in blocks:
                bitmap = q.generate_bitmap(sample=self.partitions[bid].data.values, col2idx=self.columns)
                self.blocks[bid].hits += np.sum(bitmap) / len(bitmap)

        for bid in self.blocks.keys():
            self.blocks[bid].update(with_forecast=with_forecast)

    def heat_compute(self, with_forecast=False):
        for bid in self.blocks.keys():
            self.blocks[bid].compute_temperature(forget_ratio=self.args.forget_ratio, with_forecast=with_forecast)

    def block_place(self, heat_model):
        weights = [b.size for b in self.blocks.values()]

        if heat_model == 'current':
            values = [b.temperature for b in self.blocks.values()]
        elif heat_model == 'future':
            values = [b.future_heat for b in self.blocks.values()]
        else:
            values = [b.temperature + self.args.future_weight * b.future_heat for b in self.blocks.values()]

        block_in_cache = knapsackSolver(weights=weights, values=values, capacity=self.args.cache_budget,
                                        algorithm=self.args.place_strategy)
        block_in_cloud = list(set([i for i in range(len(self.blocks))]).difference(set(block_in_cache)))

        index = [b for b in self.blocks.keys()]
        self.block_in_cache = [index[i] for i in block_in_cache]
        self.block_in_cloud = [index[i] for i in block_in_cloud]

    def supervise(self, workload_batch, with_forecast=False):
        self.update(batch=workload_batch, with_forecast=with_forecast)

    def predict(self, win_size, predict_horizon):
        return

    def migrate(self, heat_model, with_forecast=False, forecast_series=None):
        self.heat_compute(with_forecast=with_forecast)
        self.block_place(heat_model=heat_model)

    def get_blocks(self):
        return self.block_in_cache, self.block_in_cloud

