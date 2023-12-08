import random
import math


class Base(object):
    def __init__(self, args):
        self.args = args
        self.pool_size = args.num_page
        self.pool = []

    def query_route(self, query_blocks):
        blocks_hits = list(set(self.pool).intersection(set(query_blocks)))
        blocks_miss = list(set(query_blocks).difference(set(blocks_hits)))
        return blocks_hits, blocks_miss

    def update(self, query_blocks, timestamp):
        pass

    def cache(self):
        return self.pool


class FIFO(Base):
    def __init__(self, args):
        super().__init__(args)

    def update(self, query_blocks, timestamp):
        blocks_hits, blocks_miss = self.query_route(query_blocks)
        for block in blocks_miss:
            if len(self.pool) >= self.pool_size:
                self.pool.pop(0)
                self.pool.append(block)
            else:
                self.pool.append(block)


class LRU(Base):
    def __init__(self, args):
        super().__init__(args)
        self.queue = []

    def query_route(self, query_blocks):
        blocks_hits = list(set(self.pool).intersection(set(query_blocks)))
        blocks_hits += list(set(self.queue).intersection(set(query_blocks)))
        blocks_miss = list(set(query_blocks).difference(set(blocks_hits)))
        return blocks_hits, blocks_miss

    def update(self, query_blocks, timestamp):
        blocks_hits, blocks_miss = self.query_route(query_blocks)
        for block in blocks_hits:
            if block in self.queue:
                self.queue.remove(block)
                self.pool.append(block)
            elif block in self.pool:
                self.pool.remove(block)
                self.pool.append(block)
            else:
                self.queue.append(block)
                if len(self.pool) + len(self.queue) > self.pool_size:
                    self.queue.pop(0)

        for block in blocks_miss:
            if len(self.pool) + len(self.queue) >= self.pool_size:
                self.pool.pop(0)
                self.pool.append(block)
            else:
                self.pool.append(block)

    def cache(self):
        return self.pool + self.queue


class LFU(Base):
    def __init__(self, args):
        super().__init__(args)
        self.freq = []

    def update(self, query_blocks, timestamp):
        blocks_hits, blocks_miss = self.query_route(query_blocks)
        for block in blocks_hits:
            loc = self.pool.index(block)
            self.freq[loc] += 1

        for block in blocks_miss:
            if len(self.pool) >= self.pool_size:
                min_val = min(self.freq)
                loc = self.freq.index(min_val)
                self.pool.pop(loc)
                self.freq.pop(loc)
                self.pool.append(block)
                self.freq.append(1)
            else:
                self.pool.append(block)
                self.freq.append(1)


class CLOCK(Base):
    def __init__(self, args):
        super().__init__(args)
        self.check = []
        self.point = 0

    def update(self, query_blocks, timestamp):
        blocks_hits, blocks_miss = self.query_route(query_blocks)
        for block in blocks_hits:
            self.check[self.pool.index(block)] += 1

        for block in blocks_miss:
            if len(self.pool) >= self.pool_size:
                while self.check[self.point] > 1:
                    self.check[self.point] -= 1
                    self.point = int((self.point + 1) % self.pool_size)
                self.pool[self.point] = block
                self.check[self.point] = 1
                self.point = int((self.point + 1) % self.pool_size)
            else:
                self.pool.append(block)
                self.check.append(1)


class ARC(object):
    def __init__(self, args):
        self.args = args
        self.pool_size = args.num_page
        self.lru = []
        self.lfu = []
        self.freq = []
        self.lru_ghost = []
        self.lfu_ghost = []
        self.freq_ghost = []

    def query_route(self, query_blocks):
        blocks_hits = list(set(self.lru).intersection(set(query_blocks)))
        blocks_hits += list(set(self.lfu).intersection(set(query_blocks)))
        blocks_miss = list(set(query_blocks).difference(set(blocks_hits)))
        return blocks_hits, blocks_miss

    def update(self, query_blocks, timestamp):
        for block in query_blocks:

            if block in self.lru:
                self.lru.remove(block)
                self.lfu.append(block)
                self.freq.append(2)

            elif block in self.lfu:
                self.freq[self.lfu.index(block)] += 1

            elif block in self.lru_ghost:
                if len(self.lfu) > 0:
                    self.lru_ghost.remove(block)
                    self.lru.append(block)
                    min_val = min(self.freq)
                    loc = self.freq.index(min_val)
                    r_b = self.lfu.pop(loc)
                    r_f = self.freq.pop(loc)
                    self.lfu_ghost.append(r_b)
                    self.freq_ghost.append(r_f)
                else:
                    self.lru_ghost.remove(block)
                    r_b = self.lru.pop(0)
                    self.lru_ghost.append(r_b)
                    self.lru.append(block)

            elif block in self.lfu_ghost:
                if len(self.lru) > 0:
                    loc = self.lfu_ghost.index(block)
                    self.lfu.append(self.lfu_ghost[loc])
                    self.freq.append(self.freq_ghost[loc] + 1)
                    self.lfu_ghost.pop(loc)
                    self.freq_ghost.pop(loc)
                    r_b = self.lru.pop(0)
                    self.lru_ghost.append(r_b)
                else:
                    min_val = min(self.freq)
                    loc = self.freq.index(min_val)
                    r_b = self.lfu.pop(loc)
                    r_f = self.freq.pop(loc)
                    self.lfu_ghost.append(r_b)
                    self.freq_ghost.append(r_f)

                    loc = self.lfu_ghost.index(block)
                    self.lfu.append(self.lfu_ghost[loc])
                    self.freq.append(self.freq_ghost[loc] + 1)
                    self.lfu_ghost.pop(loc)
                    self.freq_ghost.pop(loc)

            else:
                if len(self.lru) + len(self.lfu) >= self.pool_size:
                    self.lru.append(block)
                    r_b = self.lru.pop(0)
                    self.lru_ghost.append(r_b)
                else:
                    self.lru.append(block)

    def cache(self):
        return self.lru + self.lfu


class LeCaR(object):
    def __init__(self, args):
        self.args = args
        self.pool_size = args.num_page
        self.w = 0.5
        self.pool = []
        self.freq = []
        self.h_lru = []
        self.h_lfu = []
        self.time_record = {}

    def query_route(self, query_blocks):
        blocks_hits = list(set(self.pool).intersection(set(query_blocks)))
        blocks_miss = list(set(query_blocks).difference(set(blocks_hits)))
        return blocks_hits, blocks_miss

    def update(self, query_blocks, timestamp):
        for block in query_blocks:

            if block in self.pool:
                loc = self.pool.index(block)
                self.pool.pop(loc)
                self.pool.append(block)
                freq = self.freq.pop(loc)
                self.freq.append(freq + 1)

                self.time_record[block] = timestamp

            else:
                if block in self.h_lru:
                    self.h_lru.remove(block)
                    method = 'lru'
                else:
                    self.h_lfu.remove(block)
                    method = 'lfu'

                t = timestamp - self.time_record[block]
                self.weight(method=method, t=t)

                if len(self.pool) < self.pool_size:
                    self.pool.append(block)
                    self.freq.append(1)
                    self.time_record[block] = timestamp

                else:
                    # assume that hist_size == pool_size
                    if random.random() < self.w:
                        # act = 'lru'
                        if len(self.h_lru) >= self.pool_size:
                            self.h_lru.pop(0)
                        r_b = self.pool.pop(0)
                        self.freq.pop(0)
                        self.h_lru.append(r_b)

                    else:
                        # act = 'lfu'
                        if len(self.h_lfu) >= self.pool_size:
                            self.h_lfu.pop(0)
                        loc = self.freq.index(min(self.freq))
                        r_b = self.pool.pop(loc)
                        self.freq.pop(loc)
                        self.h_lfu.append(r_b)

                    self.pool.append(block)
                    self.freq.append(1)
                    self.time_record[block] = timestamp

    def weight(self, method, t):
        lfu_w = 1 - self.w
        lru_w = self.w

        lc_r = math.pow(self.args.lc_d, t)

        if method == 'lru':
            lfu_w = lfu_w * math.exp(self.args.lc_p * lc_r)
        elif method == 'lfu':
            lru_w = lru_w * math.exp(self.args.lc_p * lc_r)
        else:
            raise NotImplementedError

        lru_w = lru_w / (lru_w + lfu_w)
        self.w = lru_w

    def cache(self):
        return self.pool


class TCR(object):

    class Entity(object):
        def __init__(self, args, temperature, timestamp):
            self.args = args
            self.temperature = temperature
            self.timestamp = timestamp

        def update(self, timestamp, heat_model='newton', add_hit=True):
            if heat_model == 'newton':
                if timestamp == self.timestamp:
                    if add_hit:
                        heat = self.temperature * math.exp(-self.args.trc_a * (timestamp - self.timestamp)) \
                               + self.args.tcr_init_heat
                    else:
                        heat = self.temperature
                else:
                    heat = self.temperature + self.args.tcr_init_heat
            else:
                raise NotImplementedError
            self.temperature = heat
            self.timestamp = timestamp

    def __init__(self, args):
        self.args = args
        self.pool_size = args.cache_budget / args.page_size
        self.init_heat = self.args.tcr_init_heat
        self.out_size = self.args.tcr_batch_size
        self.pool = {}
        self.history = {}

    def query_route(self, query_blocks):
        pool = self.cache()
        blocks_hits = list(set(pool).intersection(set(query_blocks)))
        blocks_miss = list(set(query_blocks).difference(set(blocks_hits)))
        return blocks_hits, blocks_miss

    def update(self, query_blocks, timestamp):
        for block in query_blocks:
            if block in self.pool:
                self.pool[block].update(timestamp)

            else:
                if len(self.pool) >= self.pool_size:
                    for k in self.pool:
                        self.pool[k].update(timestamp=timestamp, add_hit=False)
                    pool = sorted(self.pool.items(), key=lambda x: x[1].temperature)
                    phase_out_blocks = [x[0] for x in pool[:self.out_size]]
                    for r_b in phase_out_blocks:
                        self.history[r_b] = self.pool[r_b]
                        del self.pool[r_b]

                else:
                    if block in self.history:
                        self.history[block].update(timestamp=timestamp, add_hit=True)
                        self.pool[block] = self.history[block]
                        del self.history[block]
                    else:
                        entity = self.Entity(args=self.args,
                                             temperature=self.init_heat,
                                             timestamp=0)
                        entity.update(timestamp=timestamp, add_hit=True)
                        self.pool[block] = entity

    def cache(self):
        return [k for k in self.pool.keys()]


def get_model(method, args):

    if method == 'fifo':
        return FIFO(args)

    elif method == 'lru':
        return LRU(args)

    elif method == 'lfu':
        return LFU(args)

    elif method == 'clock':
        return CLOCK(args)

    elif method == 'arc':
        return ARC(args)

    elif method == 'lecar':
        return LeCaR(args)

    elif method == 'tcr':
        return TCR(args)

    else:
        raise NotImplementedError






