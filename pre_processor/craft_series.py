import random
import numpy as np


Shape_1 = {
    'cycle': [1, 4, 3, 6, 4, 2],
    'bias': (0, 2),
}


Shape_2 = {
    'cycle': [1, 2, 3, 3, 2, 1],
    'bias': (0, 2),
}


Shape_3 = {
    'cycle': [1, 1, 8, 2, 4, 1],
    'bias': (0, 2),
}


Shape_4 = {
    'cycle': [2, 2, 1, 3, 4, 3],
    'bias': (0, 2),
}


Shape_5 = {
    'cycle': [2, 4, 2, 4, 2, 4],
    'bias': (0, 2),
}


Shape_6 = {
    'cycle': [1, 1, 2, 3, 1, 1],
    'bias': (0, 1),
}


Shape = {
    '1': (0.25, Shape_1),
    '2': (0.15, Shape_2),
    '3': (0.10, Shape_3),
    '4': (0.20, Shape_4),
    '5': (0.10, Shape_5),
    '6': (0.20, Shape_6),
}


def get_scale():
    prob = random.random()
    if prob < 0.5:
        scale = np.random.uniform(1.0, 1.5)
    elif prob < 0.75:
        scale = np.random.uniform(1.5, 2.0)
    elif prob < 0.875:
        scale = np.random.uniform(2.0, 2.5)
    else:
        scale = np.random.uniform(2.5, 3.0)
    return scale


def get_shape(scale=1.0, with_offset=False, with_bias=False):
    idx = np.random.choice(a=[k for k in Shape.keys()], size=1, p=[v[0] for v in Shape.values()])[0]
    shape = Shape[idx][1]
    cycle = shape['cycle']
    cycle = [int(e * scale) for e in cycle]
    if with_offset:
        offset = random.randint(a=0, b=len(cycle))
        cycle = (cycle + cycle)[offset: offset + len(cycle)]
    if with_bias:
        bias = random.randint(a=shape['bias'][0], b=shape['bias'][1])
        cycle = [e + bias for e in cycle]
    return cycle


def series_generate(size, with_offset, with_bias):
    scale = get_scale()
    shape = get_shape(scale, with_offset, with_bias)
    num_unit = int(size / len(shape))
    series = shape
    for i in range(num_unit):
        unit = shape
        series = np.append(series, unit)
    series = series[:size]
    series[series < 0] = 0
    return series

