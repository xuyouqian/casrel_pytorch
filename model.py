import torch.nn as nn
import torch
from transformers import BertModel


class CasRel(nn.Module):
    def __init__(self, config):
        super(CasRel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_path)
        self.sub_heads_linear = nn.Linear(self.config.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.config.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(self.config.bert_dim, self.config.num_rel)
        self.obj_tails_linear = nn.Linear(self.config.bert_dim, self.config.num_rel)
        self.alpha = 0.25
        self.gamma = 2

    def get_encoded_text(self, token_ids, mask):
        encoded_text = self.bert(token_ids, attention_mask=mask)[0]
        return encoded_text

    def get_subs(self, encoded_text):
        pred_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        pred_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))
        return pred_sub_heads, pred_sub_tails

    def get_objs_for_specific_sub(self, sub_head2tail, sub_len, encoded_text):
        # sub_head_mapping [batch, 1, seq] * encoded_text [batch, seq, dim]
        sub = torch.matmul(sub_head2tail, encoded_text)  # batch size,1,dim
        sub_len = sub_len.unsqueeze(1)
        sub = sub / sub_len  # batch size, 1,dim
        encoded_text = encoded_text + sub
        #  [batch size, seq len,bert_dim] -->[batch size, seq len,relathion counts]
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))
        return pred_obj_heads, pred_obj_tails

    def forward(self, input_ids, mask, sub_head2tail, sub_len):
        """

        :param token_ids:[batch size, seq len]
        :param mask:[batch size, seq len]
        :param sub_head:[batch size, seq len]
        :param sub_tail:[batch size, seq len]
        :return:
        """
        encoded_text = self.get_encoded_text(input_ids, mask)
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)
        sub_head2tail = sub_head2tail.unsqueeze(1)  # [[batch size,1, seq len]]
        pred_obj_heads, pre_obj_tails = self.get_objs_for_specific_sub(sub_head2tail, sub_len, encoded_text)

        return {
            "pred_sub_heads": pred_sub_heads,
            "pred_sub_tails": pred_sub_tails,
            "pred_obj_heads": pred_obj_heads,
            "pred_obj_tails": pre_obj_tails,
            'mask': mask
        }

    def compute_loss(self, pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails, mask, sub_heads,
                     sub_tails, obj_heads, obj_tails):
        rel_count = obj_heads.shape[-1]
        rel_mask = mask.unsqueeze(-1).repeat(1, 1, rel_count)
        loss_1 = self.loss_fun(pred_sub_heads, sub_heads, mask)
        loss_2 = self.loss_fun(pred_sub_tails, sub_tails, mask)
        loss_3 = self.loss_fun(pred_obj_heads, obj_heads, rel_mask)
        loss_4 = self.loss_fun(pred_obj_tails, obj_tails, rel_mask)
        return loss_1 + loss_2 + loss_3 + loss_4

    def loss_fun(self, logist, label, mask):
        count = torch.sum(mask)
        logist = logist.view(-1)
        label = label.view(-1)
        mask = mask.view(-1)

        alpha_factor = torch.where(torch.eq(label, 1), 1 - self.alpha, self.alpha)
        focal_weight = torch.where(torch.eq(label, 1), 1 - logist, logist)

        loss = -(torch.log(logist) * label + torch.log(1 - logist) * (1 - label)) * mask
        return torch.sum(focal_weight * loss) / count

