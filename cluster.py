import numpy as np
import pickle
import pandas as pd
import math
from collections import defaultdict
from datasketch import MinHash, MinHashLSH
from collections import Counter


def Kmeans(data, K, maxIter):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=K, init='k-means++', max_iter=maxIter, n_init='auto')
    kmeans.fit(data)
    index = kmeans.labels_
    return index


def FBKmeans(data, K, lmbda, maxIter):
    def initialCentroid(data, K, n):
        centroid = data[np.random.choice(n, K), :]
        return centroid

    def getCentroid(data, index, K, n, m):
        centroid = np.zeros((K, m))
        for k in range(K):
            members = (index == k + 1)
            if any(members):
                centroid[k, :] = np.sum(data[members, :], axis=0) / np.sum(members)
            else:
                centroid[k, :] = data[np.random.choice(n, 1), :]

        return centroid

    def getDistance(data, centroid, K, n, m, size_cluster, lmbda):
        D = np.zeros((n, K))
        for k in range(K):
            D[:, k] = np.sum((data - centroid[k, :]) ** 2, axis=1) + lmbda * size_cluster[k]

        return D

    n, m = data.shape

    centroid = initialCentroid(data, K, n)

    size_cluster = np.ones(K)

    index = None
    sumbest = np.inf

    for i in range(maxIter):
        D = getDistance(data, centroid, K, n, m, size_cluster, lmbda)
        idx = np.argmin(D, axis=1)
        d = np.min(D, axis=1)
        totalsum = np.sum(d)
        if np.abs(sumbest - totalsum) < 1e-2:
            break
        elif totalsum < sumbest:
            index = idx
            size_cluster = np.histogram(index, bins=np.arange(K + 1))[0]
            centroid = getCentroid(data, index, K, n, m)
            sumbest = totalsum
        else:
            pass

    return index


def Propagation(data, args):
    from sklearn.cluster import AffinityPropagation
    agent = AffinityPropagation(max_iter=32, convergence_iter=8, random_state=args.seed)
    agent.fit(data)
    return agent.labels_


def AgglomerativeCluster(data, K):
    from sklearn.cluster import AgglomerativeClustering
    agent = AgglomerativeClustering(n_clusters=K, affinity='manhattan', linkage='average')
    agent.fit(data)
    return agent.labels_


def LSH(data, args):
    num_perm = args.hash_K * args.hash_L
    lsh = MinHashLSH(num_perm=num_perm, threshold=args.lsh_threshold, params=(args.hash_K, args.hash_L))
    clusters = {}
    records = {}
    cls_num = 0
    for tup_idx in range(len(data)):
        tup = data[tup_idx]
        feat = np.where(tup == 1.0)[0]
        feat = [str(i) for i in feat]
        minhash = MinHash(num_perm=num_perm)
        minhash.update_batch(p.encode('utf-8') for p in feat)
        neighbors = lsh.query(minhash)

        if len(neighbors) == 0:
            clusters[cls_num] = [tup_idx]
            records[tup_idx] = cls_num
            cls_num += 1

        else:
            neighbors = [records[int(i)] for i in neighbors]
            counter = Counter(neighbors)
            voting = {k: v * v / len(clusters[k]) for k, v in counter.items()}
            cls_idx = max(voting, key=voting.get)
            clusters[cls_idx].append(tup_idx)
            records[tup_idx] = cls_idx

        lsh.insert(str(tup_idx), minhash)

    return clusters, records


def fastKmeans(data, K):
    uni_vectors = np.unique(data, axis=0)
    centroids = uni_vectors[np.random.choice(len(uni_vectors), K), :]
    scales = np.zeros(K)
    index = []
    for row_id in range(len(data)):
        min_dist = np.inf
        assign_cls_idx = -1
        for c_id in range(K):
            dist = np.linalg.norm(centroids[c_id] - data[row_id])
            if dist < min_dist:
                min_dist = dist
                assign_cls_idx = c_id
        index.append(assign_cls_idx)
        scales[assign_cls_idx] += 1
        centroids[assign_cls_idx] = ((scales[assign_cls_idx] - 1) * centroids[assign_cls_idx]
                                     + data[row_id]) / scales[assign_cls_idx]
    return index


def nnLSH(data, args, adjust_point):
    num_perm = args.hash_K * args.hash_L
    lsh = MinHashLSH(num_perm=num_perm, threshold=args.lsh_threshold, params=(args.hash_K, args.hash_L))
    clusters = {}
    records = {}
    cls_num = 0
    for tup_idx in range(len(data)):
        tup = data[tup_idx]
        feat = np.where(tup == 1.0)[0]
        feat = [str(i) for i in feat]
        minhash = MinHash(num_perm=num_perm)
        minhash.update_batch(p.encode('utf-8') for p in feat)
        neighbors = lsh.query(minhash)

        if len(neighbors) == 0:
            clusters[cls_num] = [tup_idx]
            records[tup_idx] = cls_num
            cls_num += 1

        else:
            neighbors = [records[int(i)] for i in neighbors]
            counter = Counter(neighbors)
            voting = {k: v * v / len(clusters[k]) for k, v in counter.items()}
            cls_idx = max(voting, key=voting.get)
            clusters[cls_idx].append(tup_idx)
            records[tup_idx] = cls_idx

        lsh.insert(str(tup_idx), minhash)

        if tup_idx > 0 and tup_idx % adjust_point == 0:

            for item in records:
                tup = data[item]
                feat = np.where(tup == 1.0)[0]
                feat = [str(i) for i in feat]
                minhash = MinHash(num_perm=num_perm)
                minhash.update_batch(p.encode('utf-8') for p in feat)
                neighbors = lsh.query(minhash)
                neighbors = [records[int(i)] for i in neighbors]
                counter = Counter(neighbors)
                voting = {k: v * v / len(clusters[k]) for k, v in counter.items()}
                assign_label = max(voting, key=voting.get)
                records[item] = assign_label

    return clusters, records


def cluster(args, data, K, algorithm):
    if algorithm == 'balance_cluster':
        if args.balance_ratio == 0.0:
            index = Kmeans(data=data, K=K, maxIter=args.kmeans_epochs)
        else:
            index = FBKmeans(data=data, K=K, lmbda=args.balance_ratio, maxIter=args.kmeans_epochs)
    elif algorithm == 'k_means':
        index = Kmeans(data=data, K=K, maxIter=args.kmeans_epochs)
    elif algorithm == 'fk_means':
        index = fastKmeans(data=data, K=K)
    elif algorithm == 'lsh':
        clusters, index = LSH(data=data, args=args)
    elif algorithm == 'nn_lsh':
        clusters, index = nnLSH(data=data, args=args, adjust_point=args.adjust_cluster_point)
    elif algorithm == 'propagation':
        index = Propagation(data=data, args=args)
    elif algorithm == 'aggl':
        index = AgglomerativeCluster(data=data, K=K)
    else:
        raise NotImplementedError
    return index


def hierarchical_cluster(args, data, matrix, n_cluster, algorithm, cluster_size):
    from utility import Node
    from data_reorganization import KDTree
    Count = 0

    indexer = [i for i in range(len(data))]
    leaves = {0: Node(node_id=0, index=indexer, parent_id=-1)}

    CanSplit = True

    while CanSplit:
        CanSplit = False

        leaves_for_split = [leaf for leaf in leaves.values() if leaf.can_split]

        if len(leaves) == 0:
            break

        for leaf in leaves_for_split:

            if leaf.size < 2 * cluster_size:
                leaves[leaf.node_id].can_split = False
                continue

            if leaf.size >= n_cluster * cluster_size:
                K = n_cluster
            elif leaf.size >= int(math.sqrt(n_cluster)) * cluster_size:
                K = max(2, int(math.sqrt(n_cluster)))
            else:
                K = 2

            row_index = [idx for idx in leaf.index]
            mat = data.loc[row_index]

            uni_vectors = np.unique(mat.values, axis=0)

            if len(uni_vectors) <= K:

                nodes, Count = KDTree(data=matrix,
                                      index=leaf.index,
                                      block_size=cluster_size,
                                      root_id=Count)

                for node in nodes:
                    leaves[node.node_id] = node

            else:
                index = cluster(data=mat.values, K=K, algorithm=algorithm, args=args)

                container = defaultdict(list)
                for idx in range(len(index)):
                    container[index[idx]].append(leaf.index[idx])

                min_cluster_size = min([len(container[k]) for k in container.keys()])
                if min_cluster_size < cluster_size / 16:

                    nodes, Count = KDTree(data=matrix,
                                          index=leaf.index,
                                          block_size=cluster_size,
                                          root_id=Count)

                    for node in nodes:
                        leaves[node.node_id] = node

                else:
                    for label in container.keys():
                        Count += 1
                        node = Node(node_id=Count, index=container[label], parent_id=leaf.node_id)
                        node.set_domains(data=data.loc[node.index])
                        leaves[Count] = node

            leaves.pop(leaf.node_id)

            CanSplit = True

    clusters = []
    for leaf in leaves.values():
        cls = leaf.index
        clusters.append(cls)

    return clusters

