import json
import numpy as np
from random import seed, shuffle, random, randint, choice
from collections import defaultdict
import time


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
    assert len(ori_seq) == len(rec_seq)
    return len([ii for ii in range(len(ori_seq)) if ori_seq[ii] != rec_seq[ii]])
    # return 1 if ori_seq == rec_seq else 0


if __name__ == '__main__':
    monitor = ''

    with open('test_data/dna_seq_rebuild_val.json', encoding='utf8') as f:
        data = json.load(f)
    acc = []
    recs = []
    mutated_seqs = []
    original_seqs = []
    start_time = time.time()
    for i, data_list in enumerate(data['data']):
        # print(data_list['Original'])
        ori = data_list['dna_seq']
        rec_seq = bislide_recover(data_list['mutated_results'], len(ori))
        acc.append(cal_acc(ori, rec_seq))
        recs.append(rec_seq)

    acc = np.array(acc)
    duration = time.time() - start_time
    index = np.argmax(acc)
    print(data['data'][index], recs[index])
    print(sum(acc) / len(acc))
    print(len(acc))
    print(duration, "s")
    # with open('dna_seq_rebuild_train.json', encoding='utf8') as f:
    #     data = json.load(f)
    # acc = []
    # mutated_seqs = []
    # original_seqs = []
    # for i, data_list in enumerate(data['data']):
    #     rec = bislide_recover(data_list['mutated_results'], 139)
    #     acc.append(cal_acc(data_list['dna_seq'], rec))

    # print(sum(acc) / len(acc))
    # min_seq = min(zip(acc, data['data']), key=lambda x: x[0])[1]
    # print(min(acc))
    # print(min_seq['dna_seq'])
    # print(min_seq['mutated_results'])
    # print(bislide_recover(min_seq['mutated_results'], 139))
    # print(acc.count(1.0) / len(acc))
