__author__ = "Zhang, Haoling [zhanghaoling@genomics.cn]"

import json
import time
import numpy as np
from evaluation import DefaultCoder
from collections import defaultdict
from random import seed, shuffle, random, randint, choice
from numpy import array, zeros, arange, fromfile, packbits, unpackbits, expand_dims, concatenate, add
from numpy import log2, max, sum, ceil, where, uint8, uint64
from Chamaeleo.utils.monitor import Monitor
from Chamaeleo.utils.pipelines import TranscodePipeline

######################################## 针对Index的操作,添加聚类

def divide_all(bit_segments, index_binary_length=None, need_logs=False):
    """
    按照index归类序列
    """
    if index_binary_length is None:
        index_binary_length = int(len(str(bin(len(bit_segments)))) - 2)

    if need_logs:
        print("Divide index and data from binary matrix.")

    monitor = Monitor()
    indices_data_dict = {}

    for row in range(len(bit_segments)):
        index, data = divide(bit_segments[row], index_binary_length)

        if index not in indices_data_dict:
            indices_data_dict[index] = []

        # 针对冗余情况
        indices_data_dict[index].append(data)

        if need_logs:
            monitor.output(row + 1, len(bit_segments))

    return indices_data_dict

########################################
######################################## 自己实现的编码管道
class MyPipeline(TranscodePipeline):
    """
    自己实现编码管道，加入DNA冗余重建的过程
    【还没实现】
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

                dna_sequences = results["dna"]

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

                original_dna_sequences = copy.deepcopy(dna_sequences)
                
                ## TODO：聚类纠错
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

                if "index" in info and info["index"]:
                    if "index_length" in info:
                        indices, bit_segments = indexer.divide_all(bit_segments, info["index_length"], self.need_logs)
                    else:
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
    【还没来得及实现】
    """
    def __init__(self, yang_rule=None, yin_rule=None, virtual_nucleotide="A", max_iterations=100,
        max_ratio=0.8, faster=False, max_homopolymer=4, max_content=0.6, need_logs=False):

        super().__init__(yang_rule, yin_rule, virtual_nucleotide, max_iterations,
            max_ratio, faster, max_homopolymer, max_content, need_logs)

    def decode(self, dna_sequences):
        if self.index_length is None:
            raise ValueError("The parameter \"index_length\" is needed, "
                             + "which is used to eliminate additional random binary segments.")
        if self.total_count is None:
            raise ValueError("The parameter \"total_count\" is needed, "
                             + "which is used to eliminate additional random binary segments.")

        bit_segments = []

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

            if self.need_logs:
                self.monitor.output(sequence_index + 1, len(dna_sequences))

        remain_bit_segments = []
        for bit_segment in bit_segments:
            segment_index = int("".join(list(map(str, bit_segment[:self.index_length]))), 2)
            if segment_index < self.total_count:
                remain_bit_segments.append(bit_segment)

        return remain_bit_segments

######################################## 读入图像的操作
## author:ydh
class ImgReader:
    def __init__(self):
        self.sample_rate = None
    
    def readImg(filepath:str,down_sample:bool) -> array:
        image = cv2.imread(filepath)
        if down_sample:
            rows, cols, _channels = map(int, image.shape)
            # 调整图像尺寸
            for ratio_width in [4,3,2,1]:
                if image.shape[1] % ratio_width == 0:
                    new_width = image.shape[1] // ratio_width
                    break
            for ratio_height in [4,3,2,1]:
                if image.shape[0] % ratio_width == 0:
                    new_height = image.shape[1] // ratio_height
                    break
            downsampled_image = cv2.resize(image, (new_width, new_height))
            self.sample_rate = (ratio_width,ratio_height)
        # Encode image to PNG format in memory buffer
        encoded_img = cv2.imencode('.bmp', downsampled_image)[1]
        # Convert to NumPy array
        encoded_img_np = np.array(encoded_img)
        return encoded_img_np

    def toImg(img_array:array,blur:bool):
        # 解码
        decoded_img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        # 去噪
        if blur:
            decoded_img = cv2.medianBlur(decoded_img, 3)
        # 还原
        if self.sample_rate is not None:
            decoded_img = cv2.resize(decoded_img, (new_width*self.sample_rate[0], new_height**self.sample_rate[1]))
        return decoded_img

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
        rec_seq = forward_rec + ''.join([choice(["A", "C", "G", "T"]) for _ in range(seq_len - len(forward_rec) - len(inverse_rec))]) + inverse_rec
    else:
        rec_seq = forward_rec + inverse_rec
    return rec_seq


def cal_acc(ori_seq, rec_seq):
    """
    用于测试准确率 字符级别的f1
    author: xyc
    """
    assert len(ori_seq) == len(rec_seq)
    return len([ii for ii in range(len(ori_seq)) if ori_seq[ii]==rec_seq[ii]]) / len(ori_seq)

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
        self.address, self.payload = 12, 128
        self.supplement, self.message_number = 0, 0

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
            print("Obtain binaries from file.")
        bits = unpackbits(expand_dims(fromfile(file=input_image_path, dtype=uint8), 1), axis=1).reshape(-1)
        if need_logs:
            print("%d bits are obtained." % len(bits))

        if len(bits) % (self.payload * 2) != 0:
            self.supplement = self.payload * 2 - len(bits) % (self.payload * 2)
            if need_logs:
                print("Supplement %d bits to make sure all payload lengths are same." % self.supplement)
            bits = concatenate((bits, zeros(shape=(self.supplement,), dtype=uint8)), axis=0)
        binary_messages = bits.reshape(len(bits) // (128 * 2), (128 * 2))
        self.message_number = len(binary_messages)

        if need_logs:
            print("Insert index for each binary message.")
        byte_number = ceil(log2(len(binary_messages)) / log2(256)).astype(int)
        mould = zeros(shape=(len(binary_messages), byte_number), dtype=uint8)
        integers = arange(len(binary_messages), dtype=int)
        for index in range(byte_number):
            mould[:, -1 - index] = integers % 256
            integers //= 256
        index_matrix = unpackbits(expand_dims(mould.reshape(-1), axis=1), axis=1)
        index_matrix = index_matrix.reshape(len(binary_messages), byte_number * 8)
        unused_locations = where(sum(index_matrix, axis=0) == 0)[0]
        start_location = max(unused_locations) + 1 if len(unused_locations) > 0 else 0
        index_matrix = index_matrix[:, start_location:]
        if self.address * 2 > len(index_matrix[0]):  # use the given address length.
            expanded_matrix = zeros(shape=(len(binary_messages), self.address * 2 - len(index_matrix[0])), dtype=uint8)
            index_matrix = concatenate((expanded_matrix, index_matrix), axis=1)
        elif self.address * 2 < len(index_matrix[0]):
            raise ValueError("The address length is too short to represent all addresses.")
        binary_messages = concatenate((index_matrix, binary_messages), axis=1)

        if need_logs:
            print("Encode binary messages based on the mapping scheme.")
        digit_set = 2 * binary_messages[:, :self.address + self.payload]
        digit_set += binary_messages[:, self.address + self.payload:]
        digit_set[digit_set == 0] = ord("A")
        digit_set[digit_set == 1] = ord("C")
        digit_set[digit_set == 2] = ord("G")
        digit_set[digit_set == 3] = ord("T")
        dna_sequences = []
        for digits in digit_set:
            dna_sequences.append(digits.tostring().decode("ascii"))
            if need_logs:
                self.monitor(len(dna_sequences), len(binary_messages))

        return dna_sequences

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
        binary_messages = zeros(shape=(self.message_number, 2 * (self.address + self.payload)), dtype=uint8)
        mapping = {"A": [0, 0], "C": [0, 1], "G": [1, 0], "T": [1, 1]}
        for index, dna_sequence in enumerate(dna_sequences):
            for nucleotide_index, nucleotide in enumerate(dna_sequence[:self.address + self.payload]):
                upper, lower = mapping[nucleotide]
                binary_messages[index, nucleotide_index] = upper
                binary_messages[index, nucleotide_index + self.address + self.payload] = lower
            if need_logs:
                self.monitor(index + 1, len(dna_sequences))
        binary_messages = array(binary_messages, dtype=uint8)

        if need_logs:
            print("Sort binary messages and convert them as bits.")
        index_matrix, binary_messages = binary_messages[:, :self.address * 2], binary_messages[:, self.address * 2:]
        message_number, byte_number = len(index_matrix), ceil(len(index_matrix[0]) / 8).astype(int)
        if len(index_matrix[0]) % 8 != 0:
            expanded_matrix = zeros(shape=(message_number, 8 * byte_number - len(index_matrix[0])), dtype=uint8)
            template = concatenate((expanded_matrix, index_matrix), axis=1)
        else:
            template = index_matrix
        mould = packbits(template.reshape(message_number * byte_number, 8), axis=1).reshape(message_number, byte_number)
        orders = zeros(shape=(message_number,), dtype=uint64)
        for index in range(byte_number):
            orders = add(orders, mould[:, -1 - index] * (256 ** index), out=orders,
                         casting="unsafe")  # make up according to the byte scale.
        sorted_binary_messages = zeros(shape=(self.message_number, 2 * self.payload), dtype=uint8)
        for index, order in enumerate(orders):
            if order < len(sorted_binary_messages):
                sorted_binary_messages[order] = binary_messages[index]
            if need_logs:
                self.monitor(index + 1, len(orders))

        bits = sorted_binary_messages.reshape(-1)
        bits = bits[:-self.supplement]
        if need_logs:
            print("%d bits are retrieved." % len(bits))

        if need_logs:
            print("Save bits to the file.")
        byte_array = packbits(bits.reshape(len(bits) // 8, 8), axis=1).reshape(-1)
        byte_array.tofile(file=output_image_path)

