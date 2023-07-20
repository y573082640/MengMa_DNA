from itertools import product
import json
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, MeanShift, OPTICS
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
    features = []
    for i in range(sub_len):
        subseq = dna_seq[i * subseq_len: (i + 1) * subseq_len]
        comps = [subseq.count(nuc) / subseq_len * 10 for nuc in 'ATCG']
        features.extend(comps)
    return features


# 构建所有可能的k-mer子序列
def get_kmer_features(dna_seq, kmer_len=3):
    kmer_counts = dict()
    if kmer_len == 3:
        for kmer in kmers_3:
            kmer_counts[kmer] = dna_seq.count(kmer)
    elif kmer_len == 4:
        for kmer in kmers_4:
            kmer_counts[kmer] = dna_seq.count(kmer)
    kmer_features = list(kmer_counts.values())
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


def cluster(dna_seqs, n_cluster, copy_num, avg_dist=6.4, pca_component=None):
    """
    使用队列完成不同层次的聚类聚类法
    6.4为大多数cluster内平均最大距离,因此eps设置在[5.7,6.2]之间
    40w条数据 210秒
    :param dna_seqs: 输入的dna string数组
    :param n_cluster 聚类数量
    :param copy_num : 复制比例
    :param avg_dist 用于确定eps
    :param pca_component PCA降维法的参数，设置为None则不降维
    """

    result = []
    abnormal_result = []
    rough_threshold = 3000
    print('get dna seqs features ... ')
    seq_features = get_features(dna_seqs, pca_component, kmer_len=3, sub_len=10)
    cluster_queue = queue.Queue()
    rough_time, refine_time = 0, 0
    total_refine_sample = 0
    cluster_queue.put((avg_dist + 0.2, range(len(dna_seqs))))
    print('start queue clustering  ... ')
    round_count = 0
    while cluster_queue.empty() is False:
        # print('round {} ... '.format(round_count))
        round_count += 1
        last_iter_eps, copy_group = cluster_queue.get()
        this_iter_eps = last_iter_eps - 0.2
        num_of_cluster = int(ceil(len(copy_group) / copy_num))
        if len(copy_group) == copy_num:
            result.append(copy_group)
        elif len(copy_group) < copy_num:
            abnormal_result += copy_group
        else:
            if len(copy_group) > rough_threshold and this_iter_eps>4.0:
                # print('level2_pass... {} {}'.format(len(copy_group), this_iter_eps))
                ## 二级聚类
                start = time.time()
                abnormal_ones, label_of_indices = level2_pass(seq_features[copy_group],
                                                              index_of_seqs=copy_group,
                                                              min_samples=copy_num//2,
                                                              eps=this_iter_eps)
                end = time.time()
                rough_time += end - start
            else:
                ## 三级聚类
                # print('level3_pass... {} {}'.format(len(copy_group), num_of_cluster))
                total_refine_sample += len(copy_group)
                start = time.time()
                abnormal_ones, label_of_indices = level3_pass(seq_features[copy_group], copy_group, num_of_cluster)
                end = time.time()
                refine_time += end - start
            for k in label_of_indices:
                cluster_queue.put((this_iter_eps, k))
            abnormal_result += abnormal_ones

    abnormal_time = 0
    if n_cluster > len(result):
        pass
        # start = time.time()
        # _, final_result = level3_pass(seq_features[abnormal_result], abnormal_result, n_cluster - len(result))
        # result += final_result
        # end = time.time()
        # abnormal_time += end - start

    length_of_result = list(map(len, result))
    print("clustering results is {}/{}, length of group between {} to {}".format(
        len(result),
        n_cluster,
        min(length_of_result),
        max(length_of_result))
    )
    print('rough cost is {}s, refine cost is {}s, abnormal cost is {}s'.format(round(rough_time, 3),
                                                                               round(refine_time, 3),
                                                                               round(abnormal_time, 3)))
    print('refine sample number is {},abnormal sample number is {} '.format(total_refine_sample, len(abnormal_result)))
    ret = []
    for group in result:
        ret.append([dna_seqs[i] for i in group])
    return ret


def test_max_eps(filepath, iter_time, pca_component, copy_num):
    n_clusters, dna_seqs = get_sequences_from_fasta(filepath, copy_num=copy_num)
    t1_avg_dist, t2_avg_max_inner_dist, max_inner_dist = [], [], 0
    shuffle(dna_seqs)
    seq_features = get_features(dna_seqs, pca_component, kmer_len=3, sub_len=10)
    k_distance(seq_features, copy_num+1)
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
    percentile_index = int(0.90 * len(a_sorted))
    percentile = a_sorted[percentile_index]
    avg = np.average(a_sorted)
    mid = np.median(a_sorted)
    ## 输出
    print(mid, percentile, avg)
    return distances, indices


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
    cluster_result = cluster(dna_seqs, n_cluster, pca_component=None, avg_dist=6.34, copy_num=copy_num)
    # end = time.time()
    # print('运行时间:', end - start)
    # with open(output, 'w') as fp:
    #     json.dump({
    #         'Time': end - start,
    #         'data': cluster_result
    #     }, fp)
