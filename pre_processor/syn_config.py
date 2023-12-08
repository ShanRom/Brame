
import random
import numpy as np


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


Bias_Param_Market = {
    'freq': {
        'low': 0.6, 'mid': 0.3, 'high': 0.1
    },
    'low': {
        'center': (1, 2),
        'add_noise_prob': (0.4, 0.8),
        'noise': (1, 2)
    },
    'mid': {
        'center': (2, 4),
        'add_noise_prob': (0.4, 0.8),
        'noise': (1, 2)
    },
    'high': {
        'center': (4, 8),
        'add_noise_prob': (0.4, 0.8),
        'noise': (1, 2)
    }
}

Chaos_Param_Market = {
    'freq': {
        'low': 0.6, 'mid': 0.3, 'high': 0.1
    },
    'low': {
        'start': (0, 2),
        'min': (0, 2),
        'max': (4, 8),
        'threshold': (0.4, 0.6),
        'low_bound': 0,
        'up_bound': (1, 2)
    },
    'mid': {
        'start': (0, 4),
        'min': (0, 2),
        'max': (4, 8),
        'threshold': (0.4, 0.6),
        'low_bound': 0,
        'up_bound': (1, 2)
    },
    'high': {
        'start': (0, 16),
        'min': (0, 2),
        'max': (8, 16),
        'threshold': (0.4, 0.6),
        'low_bound': 0,
        'up_bound': (1, 4)
    }
}

Season_Param_Market = {
    'freq': {
        'low': 0.6, 'mid': 0.3, 'high': 0.1
    },
    'cycle': [4, 8, 12, 16],
    'low': {
        'scale': (1, 2),
        'center': [0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8],
        'add_noise_prob': (0.4, 0.6),
        'noise': (1, 2),
    },
    'mid': {
        'scale': (2, 4),
        'center': [0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8],
        'add_noise_prob': (0.4, 0.6),
        'noise': (1, 2),
    },
    'high': {
        'scale': (4, 8),
        'center': [0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8],
        'add_noise_prob': (0.4, 0.6),
        'noise': (1, 2),
    }
}

Cycle_Unit_Param_Market = {
    'freq': {
        'low': 0.4, 'mid': 0.3, 'high': 0.3
    },
    'low': {
        'scale': (4, 8),
        'center': [0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8],
        'width': (0.04, 0.24),
        'add_noise_prob': (0.6, 0.8),
        'noise': (1, 2)
    },
    'mid': {
        'scale': (8, 12),
        'center': [0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8],
        'width': (0.04, 0.2),
        'add_noise_prob': (0.6, 0.8),
        'noise': (1, 4)
    },
    'high': {
        'scale': (12, 16),
        'center': [0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8],
        'width': (0.04, 0.16),
        'add_noise_prob': (0.6, 0.8),
        'noise': (2, 4)
    }
}

Cycle_Param_Market = {
    'unit_num': {
        1: 0.5, 2: 0.3, 3: 0.2
    },
    'period': {
        'low': 0.6, 'mid': 0.3, 'high': 0.1
    },
    'cycle': {
        'low': [4, 6],
        'mid': [6, 8, 9, 10],
        'high': [10, 12, 15, 16]
    },
    'unit_param': Cycle_Unit_Param_Market
}

Pattern = {
    'cycle': 0.6,
    'season': 0.03,
    'chaos': 0.02,
    'cycle_bias': 0.2,
    'cycle_season': 0.10,
    'cycle_season_bias': 0.05
}


def get_pattern():
    pattern = np.random.choice(a=[k for k in Pattern.keys()], size=1, p=[v for v in Pattern.values()])
    return pattern[0]


def resolver(configs, pattern):
    if 'cycle' in pattern:
        prob = random.random()
        if prob < Cycle_Param_Market['unit_num'][1]:
            num_cycle = 1
        elif prob < Cycle_Param_Market['unit_num'][1] + Cycle_Param_Market['unit_num'][2]:
            num_cycle = 2
        else:
            num_cycle = 3
        prob = random.random()
        if prob < Cycle_Param_Market['period']['low']:
            len_cycle = 'low'
        elif prob < Cycle_Param_Market['period']['low'] + Cycle_Param_Market['period']['mid']:
            len_cycle = 'mid'
        else:
            len_cycle = 'high'
        cycle = np.random.choice(Cycle_Param_Market['cycle'][len_cycle])
        components = {}
        Unit_Param = Cycle_Param_Market['unit_param']
        for idx in range(num_cycle):
            prob = random.random()
            if prob < Unit_Param['freq']['low']:
                freq = 'low'
            elif prob < Unit_Param['freq']['low'] + Unit_Param['freq']['mid']:
                freq = 'mid'
            else:
                freq = 'high'
            scale = random.randint(Unit_Param[freq]['scale'][0], Unit_Param[freq]['scale'][1])
            center_ratio = np.random.choice(Unit_Param[freq]['center'])
            center = int(cycle * center_ratio)
            width_ratio = np.random.uniform(Unit_Param[freq]['width'][0], Unit_Param[freq]['width'][1])
            width = int(cycle * round(width_ratio, 2))
            prob = np.random.uniform(Unit_Param[freq]['add_noise_prob'][0], Unit_Param[freq]['add_noise_prob'][1])
            if random.random() < prob:
                with_noise = True
            else:
                with_noise = False
            noise = random.randint(Unit_Param[freq]['noise'][0], Unit_Param[freq]['noise'][1])
            noise = (-noise, noise)
            components['period_{}'.format(idx)] = (scale, center, width, cycle, with_noise, noise)
        unit = {
            'size': configs['size'],
            'cycle': cycle,
            'components': components
        }
        configs['cycle'] = unit

    if 'bias' in pattern:
        prob = random.random()
        if prob < Bias_Param_Market['freq']['low']:
            freq = 'low'
        elif prob < Bias_Param_Market['freq']['low'] + Bias_Param_Market['freq']['mid']:
            freq = 'mid'
        else:
            freq = 'high'
        Unit_Param = Bias_Param_Market[freq]
        center = random.randint(Unit_Param['center'][0], Unit_Param['center'][1])
        prob = np.random.uniform(Unit_Param['add_noise_prob'][0], Unit_Param['add_noise_prob'][1])
        if random.random() < prob:
            with_noise = True
        else:
            with_noise = False
        noise = random.randint(Unit_Param['noise'][0], Unit_Param['noise'][1])
        noise = (-noise, noise)
        unit = {
            'size': configs['size'],
            'center': center,
            'with_noise': with_noise,
            'noise': noise
        }
        configs['bias'] = unit

    if 'chaos' in pattern:
        prob = random.random()
        if prob < Chaos_Param_Market['freq']['low']:
            freq = 'low'
        elif prob < Chaos_Param_Market['freq']['low'] + Chaos_Param_Market['freq']['mid']:
            freq = 'mid'
        else:
            freq = 'high'
        Unit_Param = Chaos_Param_Market[freq]
        start = random.randint(Unit_Param['start'][0], Unit_Param['start'][1])
        min_val = random.randint(Unit_Param['min'][0], Unit_Param['min'][1])
        max_val = random.randint(Unit_Param['max'][0], Unit_Param['max'][1])
        threshold = np.random.uniform(Unit_Param['threshold'][0], Unit_Param['threshold'][1])
        bound = (0, random.randint(Unit_Param['up_bound'][0], Unit_Param['up_bound'][1]))
        unit = {
            'size': configs['size'],
            'start': start,
            'min': min_val,
            'max': max_val,
            'threshold': threshold,
            'bound': bound
        }
        configs['chaos'] = unit

    if 'season' in pattern:
        prob = random.random()
        if prob < Chaos_Param_Market['freq']['low']:
            freq = 'low'
        elif prob < Chaos_Param_Market['freq']['low'] + Chaos_Param_Market['freq']['mid']:
            freq = 'mid'
        else:
            freq = 'high'
        cycle = np.random.choice(Season_Param_Market['cycle'])
        Unit_Param = Season_Param_Market[freq]
        scale = random.randint(Unit_Param['scale'][0], Unit_Param['scale'][1])
        center_ratio = np.random.choice(Unit_Param['center'])
        center = int(cycle * center_ratio)
        prob = np.random.uniform(Unit_Param['add_noise_prob'][0], Unit_Param['add_noise_prob'][1])
        if random.random() < prob:
            with_noise = True
        else:
            with_noise = False
        noise = random.randint(Unit_Param['noise'][0], Unit_Param['noise'][1])
        noise = (-noise, noise)
        unit = {
            'size': configs['size'],
            'cycle': cycle,
            'scale': scale,
            'center': center,
            'with_noise': with_noise,
            'noise': noise
        }
        configs['season'] = unit

    return configs


