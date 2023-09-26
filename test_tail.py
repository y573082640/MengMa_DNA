import json
import numpy as np
from random import seed, shuffle, random, randint, choice
from collections import defaultdict, Counter
import time


def recover_seq(mutated_seqs, cur_rec_seq=''):
    # 假设4个碱基中只会发生一次突变
    three_bases_patterns = Counter([mutated_seq[:min(3, len(mutated_seq))] for mutated_seq in mutated_seqs])
    candi_common_patterns = three_bases_patterns.most_common()

    most_common_pattern = candi_common_patterns[0][0]

    min_len = min(map(len, mutated_seqs))
    if min_len < 4 or len(most_common_pattern) < 3:
        return cur_rec_seq + most_common_pattern

    four_base_patterns = Counter([mutated_seq[:4] for mutated_seq in mutated_seqs])
    most_common_pattern_4base = four_base_patterns.most_common()[0][0]
    # 多条序列突变结果一致
    if len(candi_common_patterns) > 1 and \
            candi_common_patterns[0][1] == candi_common_patterns[1][1]:
        most_common_pattern = most_common_pattern_4base[:3]

    # 三位碱基相同
    if most_common_pattern in ['AAA', 'TTT', 'CCC', 'GGG']:
        rec_seq = [mutated_seq[2:] if mutated_seq[:3] == most_common_pattern \
                       else ''.join((most_common_pattern[0], mutated_seq[3:])) for mutated_seq in mutated_seqs]
        # mutated_seq[:3] == most_common_pattern -> 可确认一对相邻碱基未突变
        # 其余情况，均可留到下一次递归再判断

        return recover_seq(rec_seq, ''.join((cur_rec_seq, most_common_pattern[:2])))

    else:
        rec_seq = [mutated_seq[1:] if mutated_seq[:3] == most_common_pattern \
                       else mutated_seq[1:]
        if mutated_seq[1] == most_common_pattern[1] \
            else mutated_seq[2:]
        if mutated_seq[2:5] == most_common_pattern_4base[1:] \
            else ''.join((most_common_pattern[1], mutated_seq[1:]))
        if mutated_seq[1:3] == most_common_pattern_4base[2:4] \
            else ''.join((most_common_pattern[1], mutated_seq[2:]))
        if mutated_seq[2:4] == most_common_pattern_4base[2:4] \
            else mutated_seq[1:]
                   for mutated_seq in mutated_seqs]
        # mutated_seq[:3] == most_common_pattern -> 未突变
        # mutated_seq[2:5] == most_common_pattern_4base[1:] -> 插入，跳过插入位
        # mutated_seq[1:3] == most_common_pattern_4base[2:4] -> 中间位删除，插入原来的中间位
        # mutated_seq[2:4] == most_common_pattern_4base[2:4] -> 中间位突变，替换原来的中间位

        return recover_seq(rec_seq, ''.join((cur_rec_seq, most_common_pattern[0])))


def bislide_recover(mutated_seqs, seq_len):
    # 在序列首尾加padding，确保第一位是对的
    forward = ['P' + mutated_seq[: seq_len // 2 + 5] for mutated_seq in mutated_seqs]
    inverse = ['P' + mutated_seq[-1: -(seq_len // 2 + 5):-1] for mutated_seq in mutated_seqs]
    forward_rec = recover_seq(forward)[1:][:seq_len // 2]
    inverse_rec = recover_seq(inverse)[1:][:seq_len - len(forward_rec)][::-1]

    # 正反两条恢复链长度小于原序列，中间插入随机碱基（保证首尾正确性）
    if len(forward_rec) + len(inverse_rec) < seq_len:
        rec_seq = forward_rec + ''.join(
            [choice('ATGC') for _ in range(seq_len - len(forward_rec) - len(inverse_rec))]) + inverse_rec
    else:
        rec_seq = ''.join((forward_rec, inverse_rec))
    return rec_seq


def cal_acc(ori_seq, rec_seq):
    assert len(ori_seq) == len(rec_seq)
    # return len([ii for ii in range(len(ori_seq)) if ori_seq[ii] != rec_seq[ii]])
    return 1 if ori_seq == rec_seq else 0


def cal_wrong(ori_seq, rec_seq):
    assert len(ori_seq) == len(rec_seq)
    return len([ii for ii in range(len(ori_seq)) if ori_seq[ii] != rec_seq[ii]])


if __name__ == '__main__':
    monitor = ''

    with open('test_data/dna_seq_rebuild_train.json', encoding='utf8') as f:
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
    wrong_idxes = []
    for idx,ret in enumerate(acc):
        if ret == 0:
            wrong_idxes.append(idx)
            print('错误：', idx, cal_wrong(recs[idx], data['data'][idx]['dna_seq']))
            # print(recs[idx],data['data'][idx]['dna_seq'],data['data'][idx]['mutated_results'])
    print(wrong_idxes)
    print('正确率：', sum(acc) / len(acc))
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
