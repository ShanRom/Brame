
import time
import numpy as np
import pandas as pd
import copy
import os
import pickle
import argparse
from structures import Column, Table
from constants import DATA_ROOT


def table_prepare(table_name, path, merge=False):
    if merge:
        df_holder = None

        for file in os.listdir(path):
            df = pd.read_csv(path / file)
            df = pd.DataFrame(df)
            if df_holder is None:
                df_holder = df
            else:
                df_holder = pd.concat([df_holder, df])

        return df_holder

    else:

        df = pd.read_csv(path / f'{table_name}.csv', index_col=None)
        return pd.DataFrame(df)


def table_process(ori_table_name, new_table_name, table, path, normalize=True, cover=False, normalize_method='norm'):
    df_holder = table

    if not os.path.exists(path):
        os.mkdir(path)

    assert normalize

    if normalize:
        processed_df = pd.DataFrame()
        for col in df_holder.columns:
            data = df_holder[col]
            if normalize_method == 'norm':
                max_val = data.max()
                min_val = data.min()
                data = (data - min_val) / (max_val - min_val)

            elif normalize_method == 'map':
                distinct_vals = np.sort(data.unique())
                n_distinct_val = len(distinct_vals)
                normalize_vals = np.linspace(start=0.0, stop=1.0, num=n_distinct_val)
                maps = dict(zip(distinct_vals, normalize_vals))
                data = data.map(maps)
            else:
                raise NotImplementedError

            processed_df[col] = data

        if len(processed_df.columns) < len(df_holder.columns):
            rename_dict = {c: 'col{}'.format(idx) for idx, c in enumerate(list(processed_df.columns))}
            processed_df.rename(columns=rename_dict, inplace=True)

        print(processed_df.head(5))

        if cover:
            processed_df.to_csv(path / f'{ori_table_name}.csv', index=False)
        else:
            processed_df.to_csv(path / f'{new_table_name}.csv', index=False)

    else:

        if cover:
            df_holder.to_csv(path / f'{ori_table_name}.csv', index=False)
        else:
            df_holder.to_csv(path / f'{new_table_name}.csv', index=False)


def meta_info_collect(table_name, source_path, target_path):
    df = pd.read_csv(source_path / f'{table_name}.csv', index_col=None)
    df = pd.DataFrame(df)
    columns = df.columns
    cols2idx = {col: idx for idx, col in enumerate(columns)}
    idx2cols = {v: k for k, v in cols2idx.items()}
    ColInfoHolder = {}
    for col in cols2idx.keys():
        max_val = df[col].max()
        min_val = df[col].min()
        distinct_val = df[col].nunique()
        vocab = np.sort(df[col].unique())
        ColInfoHolder[col] = Column(name=col,
                                    idx=cols2idx[col],
                                    min_val=min_val,
                                    max_val=max_val,
                                    n_distinct=distinct_val,
                                    vocab=vocab.astype(np.float32))

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    nrows = len(df)
    ncols = len(columns)
    table = Table(name=table_name,
                  nrows=nrows,
                  ncols=ncols,
                  columns=ColInfoHolder,
                  col2idx=cols2idx,
                  idx2col=idx2cols)

    writer = open(target_path / f'{table_name}.pkl', 'wb')
    pickle.dump(table, writer)
    writer.close()


def column_info_visualize(table_name, path):
    reader = open(path / f'{table_name}.pkl', 'rb')
    table = pickle.load(reader)
    reader.close()
    cols = table.columns
    for col in cols.values():
        print(col.name, col.min_val, col.max_val, col.n_distinct)


def load_table(table_name, path):
    reader = open(path / f'{table_name}.pkl', 'rb')
    table = pickle.load(reader)
    reader.close()
    return table


def parse_arg():
    args = argparse.ArgumentParser()
    args.add_argument('--db_name', default='power')
    args.add_argument('--ori_table_name', default='original')
    args.add_argument('--new_table_name', default='base')
    args.add_argument('--table_merge', default=False)
    args.add_argument('--normalize', default=True)
    args.add_argument('--normalized_method', choices=['norm', 'map'], default='map')
    args.add_argument('--cover', default=False)
    args.add_argument('--distinct_bound', default=6)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()

    assert args.table_merge is False
    assert args.cover is False

    if args.table_merge:
        print('--Table Prepare--')
        start_time = time.time()
        table = table_prepare(table_name=args.ori_table_name, path=DATA_ROOT / f'{args.db_name}', merge=True)
        print('Time consume : {}'.format(time.time() - start_time))

    else:
        table = table_prepare(table_name=args.ori_table_name, path=DATA_ROOT / f'{args.db_name}', merge=False)

    print('--Table Process--')
    start_time = time.time()
    table_process(ori_table_name=args.ori_table_name, new_table_name=args.new_table_name,
                  table=table, path=DATA_ROOT / f'{args.db_name}',
                  normalize=args.normalize, normalize_method=args.normalized_method, cover=args.cover)
    print('Time consume : {}'.format(time.time() - start_time))

    print('--Meta Information Collection--')
    if not args.normalize or args.cover:
        table_name = args.ori_table_name
    else:
        table_name = args.new_table_name

    start_time = time.time()
    meta_info_collect(table_name=table_name, source_path=DATA_ROOT / f'{args.db_name}',
                      target_path=DATA_ROOT / f'{args.db_name}')
    print('Time consume : {}'.format(time.time() - start_time))

    column_info_visualize(table_name=table_name, path=DATA_ROOT / f'{args.db_name}')
