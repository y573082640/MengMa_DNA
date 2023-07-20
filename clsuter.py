from itertools import product
import json
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, MeanShift, OPTICS, HDBSCAN
from sklearn.cluster import BisectingKMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from random import shuffle
import scipy.spatial.distance as dist
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import DisjointSet
from collections import defaultdict
import queue
import time
from numpy import ceil
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors

kmers_2 = [''.join(p) for p in product('ACGT', repeat=2)]
kmers_3 = ["".join(p) for p in product("ACGT", repeat=3)]
kmers_4 = ["".join(p) for p in product("ACGT", repeat=4)]


def get_sequences(file_path, cut):
    with open(file_path, 'r', encoding='UTF-8') as fp:
        data_boj = json.load(fp)

    true_k = 0
    sequences = []
    shuffle(data_boj['data'])
    for data in data_boj['data'][:cut]:
        sequences += data['mutated_results']
        true_k += 1
    shuffle(sequences)
    return true_k, sequences


def get_sequences_from_fasta(file_path, copy_num):
    ret = []
    with open(file_path, 'r', encoding='UTF-8') as fp:
        sequences = fp.readlines()
    for seq in sequences:
        if '>' in seq:
            continue
        else:
            ret.append(seq.strip())
    true_k = int(len(ret) / copy_num)
    assert len(ret) % copy_num == 0
    return true_k, ret


def get_subseq_features(dna_seq, sub_len):
    seq_len = len(dna_seq)
    subseq_len = seq_len // sub_len
    assert seq_len % sub_len != 0
    features = []
    for i in range(sub_len):
        subseq = dna_seq[i * subseq_len: (i + 1) * subseq_len]
        comps = [subseq.count(nuc) / subseq_len * 10 for nuc in 'ATCG']
        features.extend(comps)
    return features


# 构建所有可能的k-mer子序列
def get_kmer_features(dna_seq, kmer_len=2):
    # 将DNA序列分成4段
    seq_len = len(dna_seq)
    segment_len = 48
    seq_segments = [dna_seq[i:i+segment_len] for i in range(0, segment_len * 4, segment_len)]

    kmer_features = []
    for segment in seq_segments:
        kmer_counts = dict()
        if kmer_len == 3:
            for kmer in kmers_3:
                kmer_counts[kmer] = segment.count(kmer)
        elif kmer_len == 2:
            for kmer in kmers_2:
                kmer_counts[kmer] = segment.count(kmer)
        elif kmer_len == 4:
            for kmer in kmers_4:
                kmer_counts[kmer] = segment.count(kmer)
        kmer_features.extend(list(kmer_counts.values()))
    return kmer_features


# 拼接特征
def get_features(seqs, n_components, sub_len, kmer_len):
    subseq_features = np.array([get_subseq_features(seq, sub_len) for seq in seqs])
    kmer_features = np.array([get_kmer_features(seq, kmer_len) for seq in seqs])
    seq_fearures = np.concatenate((kmer_features, subseq_features), axis=1)
    seq_fearures = (seq_fearures - seq_fearures.mean(axis=0)) / seq_fearures.std(axis=0)
    # PCA
    if n_components is not None:
        pca = PCA(n_components=n_components)
        pca.fit(seq_fearures)
        seq_fearures = pca.transform(seq_fearures)
        print('所保留的n个成分各自的方差百分比:', pca.explained_variance_ratio_)
        print('所保留的n个成分各自的方差值:', pca.explained_variance_)
        print(seq_fearures.shape)
    return seq_fearures


def cal_dist(seq_fearures, all_labels):
    dist_of_all = dist.pdist(seq_fearures)
    label_of_seqs = defaultdict(list)
    for dna_seq, dna_label in zip(seq_fearures, all_labels):
        label_of_seqs[dna_label].append(dna_seq)

    sum_of_max_inner_dist = 0.0
    max_inner_dist = 0.0
    for cluster_label, seqs in label_of_seqs.items():
        vector = np.array(seqs)
        dist_of_vec = dist.pdist(vector)
        if dist_of_vec.size > 0:
            sum_of_max_inner_dist += max(dist_of_vec)
            max_inner_dist = max(max(dist_of_vec), max_inner_dist)
    return np.average(dist_of_all), sum_of_max_inner_dist / len(label_of_seqs.keys()), max_inner_dist


def divide_index_into_clusters(labels, index_of_seqs):
    assert len(labels) == len(index_of_seqs)
    label_to_seqs = defaultdict(list)
    abnormal_ones = []
    for index, label in zip(index_of_seqs, labels):
        if label == -1:
            abnormal_ones.append(index)
        else:
            label_to_seqs[label].append(index)
    return abnormal_ones, list(label_to_seqs.values())


def level2_pass(seq_features, index_of_seqs, min_samples, eps):
    """
    适用于大数样本和类别的粗分类
    """
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(seq_features)
    labels = model.labels_
    result_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    result_noise_ = list(labels).count(-1)
    print("Rough Pass Sample number: %d" % len(index_of_seqs))
    print("Estimated number of clusters: %d" % result_clusters_)
    print("Estimated number of noise points: %d" % result_noise_)
    print('-' * 20)
    return divide_index_into_clusters(labels, index_of_seqs)


def level3_pass(seq_features, index_of_seqs, n_cluster):
    """
    适用于少数样本和类别的精确分类
    """
    model = KMeans(n_clusters=n_cluster, init='k-means++', n_init=10).fit(seq_features)
    return divide_index_into_clusters(model.labels_, index_of_seqs)


def cluster_pipeline(dna_seqs, n_cluster, copy_num, pca_component=None, rough_threshold=3000):
    print('get dna seqs features ... ')
    seq_features = get_features(dna_seqs, pca_component, kmer_len=3, sub_len=7)
    min_eps, max_eps = 3, 8
    eps = min_eps
    pipeline = queue.Queue()
    pipeline.put((n_cluster, range(len(dna_seqs))))
    results = []
    print('start queue clustering  ... ')
    epoch = 0
    while pipeline.empty() is False:
        print('epoch:', epoch)
        epoch += 1
        if eps >= max_eps:
            exit(-2)
        tmp_num_cluster, remained_indices = pipeline.get()
        print('eps:{},remain number:{},cluster number:{}'.format(eps, len(remained_indices), tmp_num_cluster))
        remain_results, rets = cluster(seq_features, remained_indices, copy_num, eps, rough_threshold)
        results += rets
        if len(remain_results) > rough_threshold:
            assert len(remain_results) % copy_num == 0
            tmp_num_cluster = ceil(len(remain_results) / copy_num)
            pipeline.put((tmp_num_cluster, remain_results))
        else:
            _, final_result = level3_pass(seq_features[remain_results], remain_results, n_cluster - len(results))
            results += final_result
            break
        eps = eps + 0.25
    length_of_result = list(map(len, results))
    print("clustering results is {}/{}, length of group between {} to {}".format(
        len(results),
        n_cluster,
        min(length_of_result),
        max(length_of_result))
    )


def cluster(seq_features, indices, copy_num, avg_dist=6.4, rough_threshold=3000):
    """
    使用队列完成不同层次的聚类聚类法
    6.4为大多数cluster内平均最大距离,因此eps设置在[5.7,6.2]之间
    40w条数据 210秒
    :param seq_features: 输入的dna 特征
    :param rough_threshold 区分采用哪种聚类方法的阈值
    :param copy_num : 复制比例
    :param avg_dist 用于确定eps
    :param indices dna特征中的未分类下标
    """

    result = []
    abnormal_result = []
    cluster_queue = queue.Queue()
    rough_time, refine_time = 0, 0
    total_refine_sample = 0
    cluster_queue.put((-1, indices))
    round_count = 0
    while cluster_queue.empty() is False:
        round_count += 1
        last_iter_eps, copy_group = cluster_queue.get()
        this_iter_eps = last_iter_eps + 1
        num_of_cluster = int(ceil(len(copy_group) / copy_num))
        if len(copy_group) == copy_num:
            result.append(copy_group)
        elif len(copy_group) < copy_num:
            abnormal_result += copy_group
        else:
            if len(copy_group) > rough_threshold and this_iter_eps <= 1:
                ## 二级聚类
                start = time.time()
                abnormal_ones, label_of_indices = level2_pass(seq_features[copy_group],
                                                              index_of_seqs=copy_group,
                                                              min_samples=copy_num // 2,
                                                              eps=avg_dist)
                end = time.time()
                rough_time += end - start
            elif len(copy_group) <= rough_threshold:
                ## 三级聚类
                total_refine_sample += len(copy_group)
                start = time.time()
                abnormal_ones, label_of_indices = level3_pass(seq_features[copy_group], copy_group, num_of_cluster)
                end = time.time()
                refine_time += end - start
            else:
                abnormal_ones = copy_group
                label_of_indices = []
            for k in label_of_indices:
                cluster_queue.put((this_iter_eps + 1, k))
            abnormal_result += abnormal_ones

    print('rough cost is {}s, refine cost is {}s'.format(round(rough_time, 3), round(refine_time, 3)))
    print('refine sample number is {},abnormal sample number is {} '.format(total_refine_sample,
                                                                            len(abnormal_result)))
    return abnormal_result, result


def test_max_eps(filepath, iter_time, pca_component, copy_num):
    n_clusters, dna_seqs = get_sequences_from_fasta(filepath, copy_num=copy_num)
    t1_avg_dist, t2_avg_max_inner_dist, max_inner_dist = [], [], 0
    shuffle(dna_seqs)
    seq_features = get_features(dna_seqs, pca_component, kmer_len=3, sub_len=7)
    k_distance(seq_features, copy_num + 1)
    # index_list = range(len(seq_features))
    # for i in range(iter_time):
    #     chosen_list = np.random.choice(index_list, n_clusters * 3, False)
    #     chosen_list = seq_features[chosen_list]
    #     model = KMeans(n_clusters=n_clusters).fit(chosen_list)
    #     labels = model.labels_
    #     unique, counts = np.unique(labels, return_counts=True)
    #     print(dict(zip(unique, counts)))
    #     avg_dist, avg_max_inner_dist, tmp = cal_dist(chosen_list, labels)
    #     t1_avg_dist.append(avg_dist)
    #     t2_avg_max_inner_dist.append(avg_max_inner_dist)
    #     max_inner_dist = max(max_inner_dist, tmp)
    #
    # print("pca_component is {}, iter time is {}, result is :".format(pca_component, iter_time),
    #       np.average(t1_avg_dist),
    #       np.average(t2_avg_max_inner_dist),
    #       max_inner_dist)
    return t1_avg_dist, t2_avg_max_inner_dist, max_inner_dist


def k_distance(features, k):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(features)
    distances, indices = neigh.kneighbors(features)
    max_dist = np.squeeze(distances[:, -1])
    ## 90中位数
    a_sorted = np.sort(max_dist)
    percentile_90 = int(0.99 * len(a_sorted))
    percentile_10 = int(0.01 * len(a_sorted))
    percentile_90 = a_sorted[percentile_90]
    percentile_10 = a_sorted[percentile_10]
    avg = np.average(a_sorted)
    mid = np.median(a_sorted)
    ## 输出
    print(mid, percentile_90, percentile_10, avg)
    return mid, percentile_90, percentile_10, avg


def test_hdbscan(dna_seqs, copy_num, pca_component=None):
    print('get dna seqs features ... ')
    seq_features = get_features(dna_seqs, pca_component, kmer_len=3, sub_len=7)
    model = HDBSCAN(min_cluster_size=copy_num,
                    min_samples=copy_num // 2,
                    cluster_selection_method='eom').fit(seq_features)
    labels = model.labels_
    result_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    result_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % result_clusters_)
    print("Estimated number of noise points: %d" % result_noise_)
    print('-' * 20)


def test_kmeans(dna_seqs, n_cluster, pca_component=None):
    print('get dna seqs features ... ')
    seq_features = get_features(dna_seqs, pca_component, kmer_len=2, sub_len=15)
    model = BisectingKMeans(n_clusters=n_cluster, n_init=5, init='k-means++', bisecting_strategy='largest_cluster').fit(
        seq_features)

    labels = model.labels_
    label_dict = defaultdict(list)
    for idx, label in enumerate(labels):
        label_dict[label].append(dna_seqs[idx])
    for k, v in label_dict.items():
        if 13 >= len(v) >= 12:
            print(v)
    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))


if __name__ == "__main__":
    start = time.time()
    # 运行代码
    copy_num = 9
    filepath = 'test_data/dna_seq_rebuild_val.json'
    pasta_file = 'error/p.fasta'
    output = 'test_data/dna_seq_rebuild_val_output.json'
    # avg_dist, avg_max_inner_dist, max_inner_dist = test_max_eps(pasta_file, 5, None, copy_num)
    # print(avg_dist, avg_max_inner_dist)
    # n_cluster, dna_seqs = get_sequences('test_data/dna_seq_rebuild_train.json', cut=-1)
    n_cluster, dna_seqs = get_sequences_from_fasta(pasta_file, copy_num=copy_num)
    test_kmeans(dna_seqs, n_cluster, 'mle')
    # cluster_pipeline(dna_seqs, n_cluster, copy_num=copy_num, pca_component=None)
    # end = time.time()
    # print('运行时间:', end - start)
    # with open(output, 'w') as fp:
    #     json.dump({
    #         'Time': end - start,
    #         'data': cluster_result
    #     }, fp)
