import numpy as np
import random


def bias(config):
    series = np.full(config['size'], config['center'])
    if config['with_noise']:
        low, high = config['noise']
        noise = np.random.randint(low, high + 1, config['size'])
        series += noise
    series[series < 0] = 0
    return series


def chaos(config):
    series = np.zeros(config['size'])
    pre_val = config['start']
    for idx in range(config['size']):
        if pre_val < config['min']:
            pre_val = config['min']
        elif pre_val > config['max']:
            pre_val = config['max']
        else:
            pass

        if random.random() > config['threshold']:
            series[idx] = pre_val + np.random.randint(config['bound'][0], config['bound'][1] + 1)
        else:
            series[idx] = pre_val - np.random.randint(config['bound'][0], config['bound'][1] + 1)
        pre_val = series[idx]
    series[series < 0] = 0
    return series


def season(config):
    num_unit = int(config['size'] / config['cycle'])
    series_unit = unit(config['scale'], config['center'], None, config['cycle'], method='sin')
    series = series_unit
    for i in range(num_unit):
        if config['with_noise']:
            low, high = config['noise']
            noise = np.random.randint(low, high + 1, config['cycle'])
            series = np.append(series, series_unit + noise)
        else:
            series = np.append(series, series_unit)
    series = series[:config['size']]
    series[series < 0] = 0
    return series


def unit(scale, center, width, cycle, method):
    x = np.arange(cycle)

    if method == 'gaussian':
        width = max(1, int(width / 2))
        return scale / (np.sqrt(2 * np.pi) * width) * np.exp(-(x - center) ** 2 / (2 * width ** 2))

    elif method == 'sin':
        alpha = 2 * np.pi / cycle
        beta = center * alpha
        return scale * np.sin(alpha * x - beta) + scale

    else:
        raise NotImplementedError


def cycle(config):

    def fuse(units, size):
        series_unit = np.zeros(size)
        for k in units:
            time_series, add_noise, noise_param = units[k]
            if add_noise:
                series_unit += time_series + np.random.randint(noise_param[0], noise_param[1] + 1, size)
            else:
                series_unit += time_series
        return series_unit

    num_unit = int(config['size'] / config['cycle'])
    components = {}
    for component in config['components']:
        scale, center, width, cycle, with_noise, noise = config['components'][component]
        components[component] = (unit(scale, center, width, cycle, method='gaussian'), with_noise, noise)
    series = fuse(components, config['cycle'])

    for i in range(num_unit):
        series_unit = fuse(components, config['cycle'])
        series = np.append(series, series_unit)
    series = series[:config['size']]
    series[series < 0] = 0
    return series


def series_synthesis(config):
    series = np.zeros(config['size'])
    if 'bias' in config:
        bias_series = bias(config['bias'])
        series += bias_series
    if 'chaos' in config:
        chaos_series = chaos(config['chaos'])
        series += chaos_series
    if 'season' in config:
        season_series = season(config['season'])
        series += season_series
    if 'cycle' in config:
        cycle_series = cycle(config['cycle'])
        series += cycle_series
    series = [int(series[i]) for i in range(config['size'])]
    return series

