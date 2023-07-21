__author__ = "Zhang, Haoling [zhanghaoling@genomics.cn]"

import copy
import queue
import time
from collections import defaultdict
from datetime import datetime
from itertools import product
from random import choice
from statistics import mode

import cv2
import numpy as np
from Chamaeleo.methods.ecc import ReedSolomon,Hamming
from Chamaeleo.methods.flowed import YinYangCode, DNAFountain
from Chamaeleo.methods.inherent import base_index
from Chamaeleo.utils import data_handle, indexer
from Chamaeleo.utils.data_handle import save_model, load_model
from Chamaeleo.utils.indexer import divide
from Chamaeleo.utils.monitor import Monitor
from Chamaeleo.utils.pipelines import TranscodePipeline
from numpy import array
from numpy import ceil
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from evaluation import DefaultCoder

kmers_2 = [''.join(p) for p in product('ACGT', repeat=2)]
kmers_3 = ["".join(p) for p in product("ACGT", repeat=3)]
kmers_4 = ["".join(p) for p in product("ACGT", repeat=4)]


######################################## 针对读入的操作,添加聚类
def read_bits_from_numpy(np_array, segment_length, need_logs):
    if need_logs:
        print("Read binary matrix from numpy arr: ")

    # 将a转化为uint8类型
    a = np_array.astype(np.uint8)
    # packbits将a的每个元素转化为二进制表示并打包成uint8
    b = np.unpackbits(a)
    ori_length = len(b)
    b = b.tolist()
    matrix = []
    for index in range(0, len(b), segment_length):
        if index + segment_length < len(b):
            matrix.append(b[index: index + segment_length])
        else:
            matrix.append(b[index:] + [0] * (segment_length - len(b[index:])))

    return matrix, ori_length


######################################## 针对Index的操作,添加聚类

def get_index_of_bit_segments(bit_segments, index_binary_length=None, need_logs=False):
    """
    按照index归类序列
    """
    if index_binary_length is None:
        index_binary_length = int(len(str(bin(len(bit_segments)))) - 2)

    if need_logs:
        print("Divide index and data from binary matrix.")

    indices_data_dict = {}

    for row in range(len(bit_segments)):
        index, data = divide(bit_segments[row], index_binary_length)

        # 获取每段bit_segment的index
        indices_data_dict[row] = index

    return indices_data_dict


def my_divide_all(bit_segments, index_binary_length=None, need_logs=False, total_count=None):
    if index_binary_length is None:
        index_binary_length = int(len(str(bin(len(bit_segments)))) - 2)

    if total_count is None:
        total_count = 9999999999

    if need_logs:
        print("Divide index and data from binary matrix.")

    monitor = Monitor()
    indices = []
    divided_matrix = []

    for row in range(len(bit_segments)):
        index, data = divide(bit_segments[row], index_binary_length)
        if index > total_count:
            continue
        indices.append(index)
        divided_matrix.append(data)
        if need_logs:
            monitor.output(row + 1, len(bit_segments))

    return indices, divided_matrix


########################################
######################################## 自己实现的编码管道
def list_to_indexed_dict(lst):
    indexed_dict = {}
    for index, value in enumerate(lst):
        indexed_dict[index] = value
    return indexed_dict


def are_all_elements_same(tuple_obj):
    # Check if all elements in the tuple are the same
    return all(element == tuple_obj[0] for element in tuple_obj)


def get_dna_index_map(dna_bits_map, indices_of_bit_segments, max_bit_index):
    """
    根据两条index来给DNA分组
    :param dna_bits_map dict，key为dna序列的下标，value为Tuple(i,j)即解码出两条bits的下标i,j
    :param indices_of_bit_segments list，下标为i的bit_segment对应的index
    :param max_bit_index 解码出的index不应该超过这个范围，否则无效
    """
    index_copies_map = {}
    index_bits_map = list_to_indexed_dict(indices_of_bit_segments)

    for dna_index, bit_indexes in dna_bits_map.items():
        key = tuple(sorted([index_bits_map[item] for item in bit_indexes]))
        index_copies_map[dna_index] = key

    ret = {}
    bad_result = []
    all_num = {}

    for k, v in index_copies_map.items():
        if min(v) > max_bit_index or are_all_elements_same(v) or min(v) < 0:
            bad_result.append(v)
            continue
        if v not in ret:
            ret[v] = []
        ret[v].append(k)
        all_num[v[0]] = 1
        all_num[v[1]] = 1

    print(ret.keys())
    print(len(bad_result))
    print(len(all_num.keys()))
    return ret


class MyPipeline(TranscodePipeline):
    """
    自己实现编码管道，加入DNA冗余重建的过程
    """

    def __init__(self, **info):
        super().__init__(**info)

    def transcode(self, **info):
        if "direction" in info:
            if info["direction"] == "t_c":
                segment_length = info["segment_length"] if "segment_length" in info else 120

                self.records["payload length"] = segment_length

                if "input_path" in info:
                    bit_segments, bit_size = data_handle.read_bits_from_file(info["input_path"], segment_length,
                                                                             self.need_logs)
                elif "input_string" in info:
                    bit_segments, bit_size = data_handle.read_bits_from_str(info["input_string"], segment_length,
                                                                            self.need_logs)
                elif "input_numpy" in info:
                    bit_segments, bit_size = read_bits_from_numpy(info["input_numpy"], segment_length, self.need_logs)
                else:
                    raise ValueError("There is no digital data input here!")

                original_bit_segments = copy.deepcopy(bit_segments)

                if "index" in info and info["index"]:
                    if "index_length" in info:
                        bit_segments, index_length = indexer.connect_all(bit_segments, info["index_length"],
                                                                         self.need_logs)
                    else:
                        bit_segments, index_length = indexer.connect_all(bit_segments, None, self.need_logs)

                    self.records["index length"] = index_length
                else:
                    self.records["index length"] = 0

                if self.error_correction is not None:
                    bit_segments, error_correction_length = self.error_correction.insert(bit_segments)
                    self.records["error-correction length"] = error_correction_length
                else:
                    self.records["error-correction length"] = 0

                results = self.coding_scheme.silicon_to_carbon(bit_segments, bit_size)
                dna_sequences = results['dna']

                # dna_sequences = []
                # ## 冗余复制次数
                # print('dna seq nums:', len(dna_sequences))
                # if "dup_rate" in info and info["dup_rate"]:
                #     for s in results["dna"]:
                #         dna_sequences += [s] * info["dup_rate"]
                # else:
                #     dna_sequences = results["dna"]
                # print('dna seq nums after replicate:', len(dna_sequences))
                # ##

                self.records["information density"] = round(results["i"], 3)
                self.records["encoding runtime"] = round(results["t"], 3)

                if "output_path" in info:
                    data_handle.write_dna_file(info["output_path"], dna_sequences, self.need_logs)

                return {"bit": original_bit_segments, "dna": dna_sequences}
            elif info["direction"] == "t_s":
                if "input_path" in info:
                    dna_sequences = data_handle.read_dna_file(info["input_path"], self.need_logs)
                elif "input_string" in info:
                    dna_sequences = []
                    for index, string in enumerate(info["input_string"]):
                        dna_sequences.append(string)
                else:
                    raise ValueError("There is no digital data input here!")

                ##  进入这个函数之前要变成list
                dna_sequences = [list(i) for i in dna_sequences]
                original_dna_sequences = copy.deepcopy(dna_sequences)

                # # 冗余重建
                # # 获取dna和bit段的对应关系
                # max_bit_index = self.coding_scheme.dna_count * 2  ## 此处是针对阴阳码的 hard code
                #
                # result = self.coding_scheme.carbon_to_silicon(dna_sequences)  # dna_bits_map 记录DNA和bit段的对应关系,一对多或者一对一
                # dup_bit_segments = result['bit']
                # dna_bits_map = result['map']
                # print(len(dna_sequences), len(dup_bit_segments), len(dna_bits_map))
                #
                # # 获取纠错后的bit段
                # if self.error_correction is not None:
                #     remove_result = self.error_correction.remove(dup_bit_segments)
                #     verified_dup_bit_segments = remove_result['bit']
                #     print(remove_result['e_r'], len(remove_result['e_bit']))
                # else:
                #     verified_dup_bit_segments = dup_bit_segments
                # print(len(verified_dup_bit_segments))
                # # 获取bit段的index
                # indices_of_bit_segments, _ = indexer.divide_all(verified_dup_bit_segments,
                #                                                 info["index_length"],
                #                                                 self.need_logs)  # indices_of_bit_segments
                # print('indices_of_bit_segments len and max : ', len(indices_of_bit_segments),
                #       max(indices_of_bit_segments))
                # print('total count of bit seqs: ', max_bit_index)
                # # 记录bit段的index,一对一
                # # 根据index聚类dna,index突变的就不要了
                # index_copies_map = get_dna_index_map(dna_bits_map, indices_of_bit_segments, max_bit_index)
                # # 得到copy后重建可信dna序列
                # remove_dup_dna_seqs = []
                # print('dna seq nums before rebuild:', len(original_dna_sequences))
                # print('dna seq nums rebuild groups:', len(index_copies_map.values()))
                # for index, copies in index_copies_map.items():
                #     mutated_seqs = ["".join(original_dna_sequences[i]) for i in copies]
                #     # 众数为长度
                #     mode_val = mode([len(_) for _ in mutated_seqs])
                #     rebuild_result = bislide_recover(mutated_seqs, seq_len=mode_val)
                #     remove_dup_dna_seqs.append(list(rebuild_result))
                # # 重建结束，正式解码
                # print('dna seq nums after rebuild:', len(remove_dup_dna_seqs))
                # remove_dup_dna_seqs = copy.deepcopy(remove_dup_dna_seqs)  # 防止影响到原始的original_dna_sequences
                # results = self.coding_scheme.carbon_to_silicon(remove_dup_dna_seqs)
                results = self.coding_scheme.carbon_to_silicon(dna_sequences)
                self.records["decoding runtime"] = round(results["t"], 3)

                bit_segments = results["bit"]
                bit_size = results["s"]

                if not bit_segments:
                    self.records["error rate"] = "100.00%"
                    return {"bit": None, "dna": original_dna_sequences}

                if self.error_correction is not None:
                    verified_data = self.error_correction.remove(bit_segments)
                    bit_segments = verified_data["bit"]
                    self.records["error rate"] = str(round(verified_data["e_r"] * 100, 2)) + "%"
                    self.records["error indices"] = str(verified_data["e_i"]).replace(", ", "-") \
                        if verified_data["e_i"] != [] else None
                    self.records["error bit segments"] = str(verified_data["e_bit"]).replace(", ", "-") \
                        if verified_data["e_bit"] != [] else None
                else:
                    self.records["error rate"] = None
                    self.records["error indices"] = None
                    self.records["error bit segments"] = None

                if not bit_segments:
                    return {"bit": None, "dna": original_dna_sequences}

                # total_count = self.coding_scheme.total_count
                if "index" in info and info["index"]:
                    if "index_length" in info:
                        # indices, bit_segments = my_divide_all(bit_segments, info["index_length"], self.need_logs,
                        #                                       total_count)
                        indices, bit_segments = indexer.divide_all(bit_segments, info["index_length"], self.need_logs)
                    else:
                        # indices, bit_segments = my_divide_all(bit_segments, None, self.need_logs,
                        #                                       total_count)
                        indices, bit_segments = indexer.divide_all(bit_segments, None, self.need_logs)

                    bit_segments = indexer.sort_order(indices, bit_segments, self.need_logs)

                if "output_path" in info:
                    data_handle.write_bits_to_file(info["output_path"], bit_segments, bit_size, self.need_logs)
                elif "output_string" in info:
                    string = data_handle.write_bits_to_str(bit_segments, bit_size, self.need_logs)
                    if self.need_logs:
                        print(string)

                return {"bit": bit_segments, "dna": original_dna_sequences}
            else:
                raise ValueError("Unknown parameter \"direction\", please use \"t_c\" or \"t_s\".")
        else:
            raise ValueError("Unknown parameter \"direction\", please use \"t_c\" or \"t_s\".")


######################################## 自己实现的YinYang Code
class MyYinYangCode(YinYangCode):
    """
    自己实现阴阳编码的解码过程,目的是将DNA序列根据index聚类归类
    原版的YinYang Code有一个BUG 就是在调用纠错码恢复之前就取出Index!!!
    """

    def __init__(self, yang_rule=None, yin_rule=None, virtual_nucleotide="A", max_iterations=100,
                 max_ratio=0.8, faster=False, max_homopolymer=4, max_content=0.6, need_logs=False):
        super().__init__(yang_rule, yin_rule, virtual_nucleotide, max_iterations,
                         max_ratio, faster, max_homopolymer, max_content, need_logs)
        self.dna_count = None

    def silicon_to_carbon(self, bit_segments, bit_size):
        for bit_segment in bit_segments:
            if type(bit_segment) != list or type(bit_segment[0]) != int:
                raise ValueError("The dimension of bit matrix can only be 2!")

        self.bit_size = bit_size
        self.segment_length = len(bit_segments[0])
        start_time = datetime.now()

        if self.need_logs:
            print("The bit size of the encoded file is " + str(self.bit_size) + " bits and"
                  + " the length of final encoded binary segments is " + str(self.segment_length))

        if self.need_logs:
            print("Encode bit segments to DNA sequences by coding scheme.")

        dna_sequences = self.encode(bit_segments)

        encoding_runtime = (datetime.now() - start_time).total_seconds()

        nucleotide_count = 0
        for dna_sequence in dna_sequences:
            nucleotide_count += len(dna_sequence)

        information_density = bit_size / nucleotide_count

        return {"dna": dna_sequences, "i": information_density, "t": encoding_runtime}

    def carbon_to_silicon(self, dna_sequences):
        if self.bit_size is None:
            raise ValueError("The parameter \"bit_size\" is needed, "
                             + "which guides the number of bits reserved at the end of the digital file!")
        if self.segment_length is None:
            raise ValueError("The parameter \"segment_length\" is needed, "
                             + "which clears the information that may exist in each sequence. "
                             + "For example, assuming that the coding scheme requires an even binary segment length, "
                             + "if the inputted length is an odd number, a bit [0] is added at the end.")

        for dna_sequence in dna_sequences:
            if type(dna_sequence) != list or type(dna_sequence[0]) != str:
                raise ValueError("The dimension of nucleotide matrix can only be 2!")

        start_time = datetime.now()

        if self.need_logs:
            print("Decode DNA sequences to bit segments by coding scheme.")
        bit_segments, dna_bits_map = self.decode(dna_sequences)

        for segment_index, bit_segment in enumerate(bit_segments):
            if len(bit_segment) != self.segment_length:
                bit_segments[segment_index] = bit_segment[: self.segment_length]

        decoding_runtime = (datetime.now() - start_time).total_seconds()

        return {"bit": bit_segments, "s": self.bit_size, "t": decoding_runtime, 'map': dna_bits_map}

    def encode(self, bit_segments):
        self.index_length = int(len(str(bin(len(bit_segments)))) - 2)
        self.total_count = len(bit_segments)

        if self.faster:
            dna_sequences = self.faster_encode(bit_segments)
        else:
            dna_sequences = self.normal_encode(bit_segments)

        if self.need_logs:
            print("There are " + str(len(dna_sequences) * 2 - self.total_count)
                  + " random bit segment(s) adding for logical reliability.")

        self.dna_count = len(dna_sequences)
        return dna_sequences

    def decode(self, dna_sequences):
        """
        用于将INDEX相同的DNA序列归类
        """
        if self.index_length is None:
            raise ValueError("The parameter \"index_length\" is needed, "
                             + "which is used to eliminate additional random binary segments.")
        if self.total_count is None:
            raise ValueError("The parameter \"total_count\" is needed, "
                             + "which is used to eliminate additional random binary segments.")

        bit_segments = []
        dna_bits_map = {}
        num_of_seg = 0
        for sequence_index, dna_sequence in enumerate(dna_sequences):
            upper_bit_segment, lower_bit_segment = [], []

            support_nucleotide = self.virtual_nucleotide
            for current_nucleotide in dna_sequence:
                upper_bit = self.yang_rule[base_index[current_nucleotide]]
                lower_bit = self.yin_rule[base_index[support_nucleotide]][base_index[current_nucleotide]]
                upper_bit_segment.append(upper_bit)
                lower_bit_segment.append(lower_bit)
                support_nucleotide = current_nucleotide

            bit_segments.append(upper_bit_segment)
            bit_segments.append(lower_bit_segment)

            dna_bits_map[sequence_index] = (num_of_seg, num_of_seg + 1)
            num_of_seg += 2

            if self.need_logs:
                self.monitor.output(sequence_index + 1, len(dna_sequences))

        # if remove_random_bit_seq:
        #     remain_bit_segments = []
        #     for bit_segment in bit_segments:
        #         segment_index = int("".join(list(map(str, bit_segment[:self.index_length]))), 2)
        #         if segment_index < self.total_count:
        #             remain_bit_segments.append(bit_segment)
        #
        #     return remain_bit_segments, dna_bits_map
        # else:
        return bit_segments, dna_bits_map



######################################## 从错误中恢复的代码
## author:xyc
def recover_seq(mutated_seqs, cur_rec_seq=''):
    # 假设4个碱基中只会发生一次突变
    patterns = defaultdict(int)
    for mutated_seq in mutated_seqs:
        patterns[mutated_seq[:min(3, len(mutated_seq))]] += 1
    max_count = max(patterns.values())
    most_common_pattern = max(patterns, key=patterns.get)

    min_len = min(map(len, mutated_seqs))
    if min_len < 4 or len(most_common_pattern) < 3:
        return cur_rec_seq + most_common_pattern

    common_patterns = [p for p in patterns.keys() if patterns[p] == max_count]
    # 多条序列突变结果一致
    if len(common_patterns) > 1:
        common_patterns = [mutated_seq[:4] for mutated_seq in mutated_seqs if patterns[mutated_seq[:3]] == max_count]
        # 多比较一位
        Four_base_patterns = {cc: common_patterns.count(cc) for cc in common_patterns}
        most_common_pattern = max(Four_base_patterns, key=Four_base_patterns.get)[:3]

    rec_seq = []
    # 三位碱基相同
    if most_common_pattern in ['AAA', 'TTT', 'CCC', 'GGG']:
        for mutated_seq in mutated_seqs:
            # 可确认一对相邻碱基未突变
            if mutated_seq[:3] == most_common_pattern:
                rec_seq.append(mutated_seq[2:])
            # 删除 / 末位突变，可在下一次递归中判断
            elif mutated_seq[1] == most_common_pattern[1]:
                rec_seq.append(mutated_seq[1:])
            # 插入
            elif mutated_seq[2:4] == most_common_pattern[1:3]:
                rec_seq.append(mutated_seq[2:])
            else:
                rec_seq.append(mutated_seq[2:])

        return recover_seq(rec_seq, cur_rec_seq + most_common_pattern[:2])

    else:
        for mutated_seq in mutated_seqs:
            middle_base = most_common_pattern[1]
            if mutated_seq[:3] == most_common_pattern:
                rec_seq.append(mutated_seq[1:])
            else:
                if mutated_seq[1] == most_common_pattern[1]:
                    rec_seq.append(mutated_seq[1:])
                # 中间位删除 -> 插入原来的中间位
                elif mutated_seq[1] == most_common_pattern[2]:
                    rec_seq.append(middle_base + mutated_seq[1:])
                # 中间位突变 -> 替换原来的中间位
                elif mutated_seq[2] == most_common_pattern[2]:
                    rec_seq.append(middle_base + mutated_seq[2:])
                # 插入 -> 跳过插入位
                elif mutated_seq[2: 4] == most_common_pattern[1:]:
                    rec_seq.append(mutated_seq[2:])
                else:
                    rec_seq.append(mutated_seq[1:])
        return recover_seq(rec_seq, cur_rec_seq + most_common_pattern[0])


def bislide_recover(mutated_seqs, seq_len):
    # 在序列首尾加padding，确保第一位是对的
    forward = ['P' + mutated_seq[: seq_len // 2 + 10] for mutated_seq in mutated_seqs]
    inverse = ['P' + mutated_seq[-1: -(seq_len // 2 + 10):-1] for mutated_seq in mutated_seqs]
    forward_rec = recover_seq(forward)[1:][:seq_len // 2]
    inverse_rec = recover_seq(inverse)[1:][:seq_len // 2 + seq_len % 2][::-1]

    # 正反两条恢复链长度小于原序列，中间插入随机碱基（保证首尾正确性）
    if len(forward_rec) + len(inverse_rec) < seq_len:
        rec_seq = forward_rec + ''.join(
            [choice(["A", "C", "G", "T"]) for _ in range(seq_len - len(forward_rec) - len(inverse_rec))]) + inverse_rec
    else:
        rec_seq = forward_rec + inverse_rec
    return rec_seq


def cal_acc(ori_seq, rec_seq):
    """
    用于测试准确率 字符级别的f1
    author: xyc
    """
    assert len(ori_seq) == len(rec_seq)
    return len([ii for ii in range(len(ori_seq)) if ori_seq[ii] == rec_seq[ii]]) / len(ori_seq)


######################################## 聚类算法相关

def get_subseq_features(dna_seq, sub_len):
    seq_len = len(dna_seq)
    subseq_len = seq_len // sub_len
    features = []
    for i in range(sub_len):
        subseq = dna_seq[i * subseq_len: (i + 1) * subseq_len]
        comps = [subseq.count(nuc) / subseq_len * 10 for nuc in 'ATCG']
        features.extend(comps)
    return features


def get_kmer_features(dna_seq, kmer_len=2):
    kmer_counts = dict()
    for kmer in kmers_3:
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
        print('所保留的n个成分各自的方差百分比:', pca.explained_variance_ratio_)
        print('所保留的n个成分各自的方差值:', pca.explained_variance_)
        print(seq_fearures.shape)
    return seq_fearures


def k_distance(features, k):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(features)
    distances, indices = neigh.kneighbors(features)
    max_dist = np.squeeze(distances[:, -1])
    ## 90中位数
    a_sorted = np.sort(max_dist)
    percentile_995 = int(0.995 * len(a_sorted))
    percentile_001 = int(0.001 * len(a_sorted))
    percentile_995 = a_sorted[percentile_995]
    percentile_001 = a_sorted[percentile_001]
    avg = np.average(a_sorted)
    mid = np.median(a_sorted)
    ## 输出
    print(mid, percentile_995, percentile_001, avg)
    return mid, percentile_995, percentile_001, avg


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
    mid, percentile_995, percentile_001, avg = k_distance(seq_features, copy_num)
    eps, min_eps = max(np.ceil(percentile_995), percentile_995 + 0.4), np.floor(
        percentile_001)  ## cluster_pipeline只适用于喷泉码结果，阴阳码结果聚类密度差距过大，该方法不适用
    pipeline = queue.Queue()
    pipeline.put((n_cluster, range(len(dna_seqs))))
    results = []
    print('start queue clustering  ... ')
    epoch = 0

    while pipeline.empty() is False:
        print('cluster round:', epoch)
        epoch += 1
        if eps <= min_eps:
            break
        tmp_num_cluster, remained_indices = pipeline.get()
        print('eps:{},remain number:{},cluster number:{}'.format(eps, len(remained_indices), tmp_num_cluster))
        remain_results, rets = cluster(seq_features, remained_indices, copy_num, eps, rough_threshold)
        results += rets
        if len(remain_results) > rough_threshold:
            assert len(remain_results) % copy_num == 0
            tmp_num_cluster = ceil(len(remain_results) / copy_num)
            pipeline.put((tmp_num_cluster, remain_results))
        elif n_cluster > len(results):
            _, final_result = level3_pass(seq_features[remain_results], remain_results, n_cluster - len(results))
            results += final_result
            break
        eps = eps - 0.2

    while pipeline.empty() is False:
        tmp_num_cluster, remained_indices = pipeline.get()
        _, final_result = level3_pass(seq_features[remained_indices], remained_indices, tmp_num_cluster)
        results += final_result

    length_of_result = list(map(len, results))
    print("clustering results is {}/{}, length of group between {} to {}".format(
        len(results),
        n_cluster,
        min(length_of_result),
        max(length_of_result))
    )

    ret = []
    for group in results:
        ret.append([dna_seqs[i] for i in group])
    return ret


def cluster(seq_features, indices, copy_num, avg_dist, rough_threshold):
    """
    使用队列完成不同层次的聚类聚类法
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


######################################## 读入图像的操作
## author:ydh
class ImgReader:
    def __init__(self,encode_format):
        self.sample_rate = None
        self.encode_format = encode_format

    def readImg(self, filepath: str, down_sample: bool) -> array:
        image = cv2.imread(filepath)
        if down_sample:
            # 调整图像尺寸
            for ratio_width in [4, 3, 2, 1]:
                if image.shape[1] % ratio_width == 0:
                    new_width = image.shape[1] // ratio_width
                    break
            for ratio_height in [4, 3, 2, 1]:
                if image.shape[0] % ratio_width == 0:
                    new_height = image.shape[0] // ratio_height
                    break
            downscaledimage = cv2.resize(image, (new_width, new_height))
            self.sample_rate = (ratio_width, ratio_height)
        else:
            downscaledimage = image
        # Encode image to PNG format in memory buffer
        encoded_img = cv2.imencode(self.encode_format, downscaledimage)[1]
        # Convert to NumPy array
        encoded_img_np = np.array(encoded_img)
        return encoded_img_np

    def toImg(self, img_array: array, blur: bool):
        # 解码
        decoded_img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        rows, cols, _channels = map(int, decoded_img.shape)
        # 去噪
        if blur:
            decoded_img = cv2.medianBlur(decoded_img, 3)
        # 还原
        print(self.sample_rate)
        if self.sample_rate is not None:
            decoded_img = cv2.resize(decoded_img, (cols * self.sample_rate[0], rows * self.sample_rate[1]))
        return decoded_img

######################################## 需要我们自己实现的Coder

class Coder(DefaultCoder):

    def __init__(self, team_id: str = "none"):
        """
        Initialize the image-DNA coder.

        :param team_id: team id provided by sponsor.
        :type team_id: str

        .. note::
            The competition process is automatically created.

            Thus,
            (1) Please do not add parameters other than "team_id".
                All parameters should be declared directly in this interface instead of being passed in as parameters.
                If a parameter depends on the input image, please assign its value in the "image_to_dna" interface.
            (2) Please do not divide "coder.py" into multiple script files.
                Only the script called "coder.py" will be automatically copied by
                the competition process to the competition script folder.

        """
        super().__init__(team_id=team_id)
        self.address, self.payload = 20, 160
        self.check_size = 4
        self.supplement, self.message_number = 0, 0
        self.copy_num, self.n_cluster = 9, None
        self.seq_len = Hamming()
        self.coding_scheme = DNAFountain(redundancy=2)
        self.error_correction = Hamming()
        self.reader = ImgReader(encode_format='.jpg')

    def image_to_dna(self, input_image_path, need_logs=True):
        """
        Convert an image into a list of DNA sequences.

        :param input_image_path: path of the image to be encoded.
        :type input_image_path: str

        :param need_logs: print process logs if required.
        :type need_logs: bool

        :return: a list of DNA sequences.
        :rtype: list

        .. note::
            Each DNA sequence is suggested to carry its address information in the sequence list.
            Because the DNA sequence list obtained in DNA sequencing is inconsistent with the existing list.
        """
        if need_logs:
            print("init models ...")
        pipeline = MyPipeline(coding_scheme=self.coding_scheme,
                              error_correction=self.error_correction,
                              need_logs=True)
        if need_logs:
            print("read img ...")
        img_array = self.reader.readImg(input_image_path, False)
        if need_logs:
            print("transcode bits ...")
        results = pipeline.transcode(direction="t_c",
                                     input_numpy=img_array,
                                     output_path='target.dna',
                                     segment_length=self.payload,
                                     index_length=self.address, index=True)
        dna_sequences = ["".join(_) for _ in results['dna']]
        self.n_cluster = len(dna_sequences)
        self.seq_len = len(dna_sequences[0])
        return dna_sequences * self.copy_num

    def dna_to_image(self, dna_sequences, output_image_path, need_logs=True):
        """
        Convert a list of DNA sequences to an image.

        :param dna_sequences: a list of DNA sequences (obtained from DNA sequencing).
        :type dna_sequences: list

        :param output_image_path: path for storing image data.
        :type output_image_path: str

        :param need_logs: print process logs if required.
        :type need_logs: bool

        .. note::
           The order of the samples in this DNA sequence list input must be different from
           the order of the samples output by the "image_to_dna" interface.
        """
        if need_logs:
            print("Decode DNA sequences based on the mapping scheme.")
        grouped_dna_sequences = cluster_pipeline(dna_sequences, self.n_cluster, self.copy_num, None)
        rebuild_dna_sequences = [bislide_recover(g, self.seq_len) for g in grouped_dna_sequences]
        pipeline = MyPipeline(coding_scheme=self.coding_scheme,
                              error_correction=self.error_correction,
                              need_logs=True)
        result = pipeline.transcode(direction="t_s",
                                    segment_length=self.payload,
                                    index_length=self.address,
                                    input_string=rebuild_dna_sequences,
                                    index=True)
        img_np = np.array(result['bit'])
        img_np = np.packbits(img_np)
        img = self.reader.toImg(img_np, blur=False)
        print(img.shape)
        cv2.imwrite(output_image_path, img)
