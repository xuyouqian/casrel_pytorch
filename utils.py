from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from random import choice

import json
import torch


def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    text = batch[0]
    triple = batch[1]
    del batch
    return text, triple


class MyDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.dataset = []
        with open(path, encoding='utf8') as F:
            for line in F:
                line = json.loads(line)
                self.dataset.append(line)

    def __getitem__(self, item):
        content = self.dataset[item]
        text = content['text']
        spo_list = content['spo_list']
        return text, spo_list

    def __len__(self):
        return len(self.dataset)


def create_data_iter(config):
    train_data = MyDataset(config.train_data_path)
    dev_data = MyDataset(config.dev_data_path)
    test_data = MyDataset(config.test_data_path)

    train_iter = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    test_iter = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_iter, dev_iter, test_iter


class Batch:
    def __init__(self, config):
        self.tokenizer = config.tokenizer
        self.num_relations = config.num_rel
        self.rel_vocab = config.rel_vocab
        self.device = config.device

    def __call__(self, text, triple):
        text = self.tokenizer(text, padding=True).data
        batch_size = len(text['input_ids'])
        seq_len = len(text['input_ids'][0])
        sub_head = []
        sub_tail = []
        sub_heads = []
        sub_tails = []
        obj_heads = []
        obj_tails = []
        sub_len = []
        sub_head2tail = []

        for batch_index in range(batch_size):
            inner_input_ids = text['input_ids'][batch_index]  # 单个句子变成索引后
            inner_triples = triple[batch_index]
            inner_sub_heads, inner_sub_tails, inner_sub_head, inner_sub_tail, inner_sub_head2tail, inner_sub_len, inner_obj_heads, inner_obj_tails = \
                self.create_label(inner_triples, inner_input_ids, seq_len)
            sub_head.append(inner_sub_head)
            sub_tail.append(inner_sub_tail)
            sub_len.append(inner_sub_len)
            sub_head2tail.append(inner_sub_head2tail)
            sub_heads.append(inner_sub_heads)
            sub_tails.append(inner_sub_tails)
            obj_heads.append(inner_obj_heads)
            obj_tails.append(inner_obj_tails)

        input_ids = torch.tensor(text['input_ids']).to(self.device)
        mask = torch.tensor(text['attention_mask']).to(self.device)
        sub_head = torch.stack(sub_head).to(self.device)
        sub_tail = torch.stack(sub_tail).to(self.device)
        sub_heads = torch.stack(sub_heads).to(self.device)
        sub_tails = torch.stack(sub_tails).to(self.device)
        sub_len = torch.stack(sub_len).to(self.device)
        sub_head2tail = torch.stack(sub_head2tail).to(self.device)
        obj_heads = torch.stack(obj_heads).to(self.device)
        obj_tails = torch.stack(obj_tails).to(self.device)

        return {
                   'input_ids': input_ids,
                   'mask': mask,
                   'sub_head2tail': sub_head2tail,
                   'sub_len': sub_len
               }, {
                   'sub_heads': sub_heads,
                   'sub_tails': sub_tails,
                   'obj_heads': obj_heads,
                   'obj_tails': obj_tails
               }

    def create_label(self, inner_triples, inner_input_ids, seq_len):

        inner_sub_heads, inner_sub_tails = torch.zeros(seq_len), torch.zeros(seq_len)
        inner_sub_head, inner_sub_tail = torch.zeros(seq_len), torch.zeros(seq_len)
        inner_obj_heads = torch.zeros((seq_len, self.num_relations))
        inner_obj_tails = torch.zeros((seq_len, self.num_relations))
        inner_sub_head2tail = torch.zeros(seq_len)  # 随机抽取一个实体，从开头一个词到末尾词的索引

        # 因为数据预处理代码还待优化,会有不存在关系三元组的情况，
        # 初始化一个主词的长度为1，即没有主词默认主词长度为1，
        # 防止零除报错,初始化任何非零数字都可以，没有主词分子是全零矩阵
        inner_sub_len = torch.tensor([1], dtype=torch.float)
        # 主词到谓词的映射
        s2ro_map = defaultdict(list)
        for inner_triple in inner_triples:

            inner_triple = (
                self.tokenizer(inner_triple['subject'], add_special_tokens=False)['input_ids'],
                self.rel_vocab.to_index(inner_triple['predicate']),
                self.tokenizer(inner_triple['object'], add_special_tokens=False)['input_ids']
            )

            sub_head_idx = self.find_head_idx(inner_input_ids, inner_triple[0])
            obj_head_idx = self.find_head_idx(inner_input_ids, inner_triple[2])

            if sub_head_idx != -1 and obj_head_idx != -1:
                sub = (sub_head_idx, sub_head_idx + len(inner_triple[0]) - 1)
                # s2ro_map保存主语到谓语的映射
                s2ro_map[sub].append(
                    (obj_head_idx, obj_head_idx + len(inner_triple[2]) - 1, inner_triple[1]))  # {(3,5):[(7,8,0)]} 0是关系

        if s2ro_map:
            for s in s2ro_map:
                inner_sub_heads[s[0]] = 1
                inner_sub_tails[s[1]] = 1

            sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
            inner_sub_head[sub_head_idx] = 1
            inner_sub_tail[sub_tail_idx] = 1
            inner_sub_head2tail[sub_head_idx:sub_tail_idx + 1] = 1
            inner_sub_len = torch.tensor([sub_tail_idx + 1 - sub_head_idx], dtype=torch.float)
            for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                inner_obj_heads[ro[0]][ro[2]] = 1
                inner_obj_tails[ro[1]][ro[2]] = 1

        return inner_sub_heads, inner_sub_tails, inner_sub_head, inner_sub_tail, inner_sub_head2tail, inner_sub_len, inner_obj_heads, inner_obj_tails

    @staticmethod
    def find_head_idx(source, target):
        target_len = len(target)
        for i in range(len(source)):
            if source[i: i + target_len] == target:
                return i
        return -1
