from transformers import BertTokenizer, AdamW
from model import CasRel
from tqdm import tqdm

import pandas as pd
import torch


def load_model(config):
    device = config.device
    model = CasRel(config)
    model.to(device)

    # prepare optimzier
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=10e-8)
    sheduler = None

    return model, optimizer, sheduler, device


def train_epoch(model, train_iter, dev_iter, optimizer, batch, best_triple_f1, epoch):
    for step, (text, triple) in enumerate(train_iter):
        model.train()
        inputs, labels = batch(text, triple)
        logist = model(**inputs)
        loss = model.compute_loss(**logist, **labels)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 1:
            sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df = test(model, dev_iter,
                                                                                                     batch)
            if triple_f1 > best_triple_f1:
                best_triple_f1 = triple_f1
                torch.save(model.state_dict(), 'best_f1.pth')
                print(
                    'epoch:{},step:{},sub_precision:{:.4f}, sub_recall:{:.4f}, sub_f1:{:.4f}, triple_precision:{:.4f}, triple_recall:{:.4f}, triple_f1:{:.4f},train loss:{:.4f}'.format(
                        epoch, step, sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1,
                        loss.item()))
                print(df)

    return best_triple_f1


def train(model, train_iter, dev_iter, optimizer, config, batch):
    epochs = config.epochs
    best_triple_f1 = 0
    for epoch in range(epochs):
        best_triple_f1 = train_epoch(model, train_iter, dev_iter, optimizer, batch, best_triple_f1, epoch)


def test(model, dev_iter, batch):
    model.eval()
    df = pd.DataFrame(columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'], index=['sub', 'triple'])
    df.fillna(0, inplace=True)

    for text, triple in tqdm(dev_iter):
        inputs, labels = batch(text, triple)
        logist = model(**inputs)

        pred_sub_heads = convert_score_to_zero_one(logist['pred_sub_heads'])
        pred_sub_tails = convert_score_to_zero_one(logist['pred_sub_tails'])

        sub_heads = convert_score_to_zero_one(labels['sub_heads'])
        sub_tails = convert_score_to_zero_one(labels['sub_tails'])
        batch_size = inputs['input_ids'].shape[0]

        obj_heads = convert_score_to_zero_one(labels['obj_heads'])
        obj_tails = convert_score_to_zero_one(labels['obj_tails'])
        pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])
        pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])

        for batch_index in range(batch_size):
            pred_subs = extract_sub(pred_sub_heads[batch_index].squeeze(), pred_sub_tails[batch_index].squeeze())
            true_subs = extract_sub(sub_heads[batch_index].squeeze(), sub_tails[batch_index].squeeze())

            pred_ojbs = extract_obj_and_rel(pred_obj_heads[batch_index], pred_obj_tails[batch_index])
            true_objs = extract_obj_and_rel(obj_heads[batch_index], obj_tails[batch_index])

            df['PRED']['sub'] += len(pred_subs)
            df['REAL']['sub'] += len(true_subs)
            for true_sub in true_subs:
                if true_sub in pred_subs:
                    df['TP']['sub'] += 1

            df['PRED']['triple'] += len(pred_ojbs)
            df['REAL']['triple'] += len(true_objs)
            for true_obj in true_objs:
                if true_obj in pred_ojbs:
                    df['TP']['triple'] += 1

    df.loc['sub', 'p'] = df['TP']['sub'] / (df['PRED']['sub'] + 1e-9)
    df.loc['sub', 'r'] = df['TP']['sub'] / (df['REAL']['sub'] + 1e-9)
    df.loc['sub', 'f1'] = 2 * df['p']['sub'] * df['r']['sub'] / (df['p']['sub'] + df['r']['sub'] + 1e-9)

    sub_precision = df['TP']['sub'] / (df['PRED']['sub'] + 1e-9)
    sub_recall = df['TP']['sub'] / (df['REAL']['sub'] + 1e-9)
    sub_f1 = 2 * sub_precision * sub_recall / (sub_precision + sub_recall + 1e-9)

    df.loc['triple', 'p'] = df['TP']['triple'] / (df['PRED']['triple'] + 1e-9)
    df.loc['triple', 'r'] = df['TP']['triple'] / (df['REAL']['triple'] + 1e-9)
    df.loc['triple', 'f1'] = 2 * df['p']['triple'] * df['r']['triple'] / (
            df['p']['triple'] + df['r']['triple'] + 1e-9)

    triple_precision = df['TP']['triple'] / (df['PRED']['triple'] + 1e-9)
    triple_recall = df['TP']['triple'] / (df['REAL']['triple'] + 1e-9)
    triple_f1 = 2 * triple_precision * triple_recall / (
            triple_precision + triple_recall + 1e-9)

    return sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df


def extract_sub(pred_sub_heads, pred_sub_tails):
    subs = []
    heads = torch.arange(0, len(pred_sub_heads))[pred_sub_heads == 1]
    tails = torch.arange(0, len(pred_sub_tails))[pred_sub_tails == 1]

    for head, tail in zip(heads, tails):
        if tail >= head:
            subs.append((head.item(), tail.item()))
    return subs


def extract_obj_and_rel(obj_heads, obj_tails):
    obj_heads = obj_heads.T
    obj_tails = obj_tails.T
    rel_count = obj_heads.shape[0]
    obj_and_rels = []  # [(rel_index,strart_index,end_index),(rel_index,strart_index,end_index)]

    for rel_index in range(rel_count):
        obj_head = obj_heads[rel_index]
        obj_tail = obj_tails[rel_index]

        objs = extract_sub(obj_head, obj_tail)
        if objs:
            for obj in objs:
                start_index, end_index = obj
                obj_and_rels.append((rel_index, start_index, end_index))
    return obj_and_rels


def convert_score_to_zero_one(tensor):
    tensor[tensor >= 0.5] = 1
    tensor[tensor < 0.5] = 0
    return tensor
