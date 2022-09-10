# --coding:utf-8--
import time

import torch as th
import torch.nn.functional as F
import numpy as np
import winnt
import torch.nn as nn
from torch import tensor
from torch.nn.utils.rnn import pack_padded_sequence

from DRL_rec.Stacking_SAB import stacking_SAB,feed_forward_layer,SelfAttention


class RNN(nn.Module):
    def __init__(self,in_feat=50,hid_feat=50,number=1):
        super(RNN, self).__init__()
        self.in_feat = in_feat
        self.number = number
        self.rnn = nn.GRU(self.in_feat, self.in_feat, 1, batch_first=True).cuda()
        self.rnn.reset_parameters()
    def forward(self,inputs):
        """
        :param inputs: 输入张量     tensor(item_num,e_dim)
        :return
        """
        if th.is_tensor(inputs):
            if len(inputs.shape) == 2:
                inputs = inputs.unsqueeze(dim=0)
        out, h = self.rnn(inputs)
        return h

def pad_sequence(interaction_historys,z,item_emb,mask_emb,item_gemb=None):
    """将list(batch_size,tensor(l,e_dim))补齐，其中l不相等"""
    users_items = interaction_historys[z]  # list(batch_size,list(l))
    seq_lens = [len(list) for list in users_items]
    # users_items = [list for list in users_items if len(list) > 0]
    users_items = [th.tensor(list) for list in users_items]
    users_items_pad = th.nn.utils.rnn.pad_sequence(users_items, batch_first=True, padding_value=0)
    users_items_pad = users_items_pad.to(th.long).cuda()
    if (item_gemb == None):
        users_items_emb = item_emb(users_items_pad)
    else:
        users_items_emb = item_gemb[users_items_pad]
    zeros = th.zeros_like(users_items_emb)
    mask = mask_emb(users_items_pad)
    users_items_emb = th.where(th.eq(mask, 0), zeros, users_items_emb)
    return users_items_emb, seq_lens


def pad_sequence1(interaction_historys,item_emb,mask_emb):
    """将list(batch——size,tensor(l,e_dim))补齐，其中l不相等"""
    users_items = interaction_historys  # list(batch_size,list(l))
    seq_lens = [len(list) for list in users_items]
    users_items = [list for list in users_items if len(list) > 0]
    users_items = [th.tensor(list) for list in users_items]
    users_items_pad = th.nn.utils.rnn.pad_sequence(users_items, batch_first=True, padding_value=0)
    users_items_pad = users_items_pad.to(th.long).cuda()
    users_items_emb = item_emb(users_items_pad)
    zeros = th.zeros_like(users_items_emb)
    mask = mask_emb(users_items_pad)
    users_items_emb = th.where(th.eq(mask, 0), zeros, users_items_emb)
    return users_items_emb,seq_lens

def pad_tensors(tensors):
    """
    :param tensors: [batch_size * tensor(item_num,emb_dim)]
    :return: tensor(bs,item_num_pad,d)       tensor(list[narray])
    """
    item_num = []
    tensors_pad_list = []
    for tensor in tensors:
        item_num.append(tensor.shape[0])
    max_num = max(item_num)
    for tensor in tensors:
        dim = (0,0,0,max_num-tensor.shape[0])
        tensor = F.pad(tensor,dim)
        tensors_pad_list.append(tensor)
    tensor_pad = th.stack(tensors_pad_list,dim=0)
    return tensor_pad

def translist2dict(users_interaction_history,dataset='explicit'):                            #list(batch_size*dict(6,list(l)))

    batch_size = len(users_interaction_history)
    empty_flag = True
    interaction_historys = {}
    if dataset == 'explicit':
        pass
    if dataset == 'implicit':
        for i in range(0,2):
            interaction_historys[i] = []
            for user_interation_history in users_interaction_history:
                if len(user_interation_history[i]) != 0:
                    empty_flag = False
                interaction_historys[i].append(user_interation_history[i])
    return batch_size,interaction_historys,empty_flag                    #dict(6,list(batch_size,l))

def get_users_perference(users_interaction_history,item_emb,mask_emb,ssab,dataset='explicit'):
    """
    :param users_interation_history: batch用户的交互历史         list(batch_size*dict(6,list(l)))
    :param item_emb:所有item的embedding字典
    :param ssab:ssab实例
    :return:
    """
    e_dim = item_emb.weight.shape[1]
    batch_size, interaction_historys,empty_flag = translist2dict(users_interaction_history=users_interaction_history,dataset=dataset)
    if empty_flag == True:
        return th.zeros((batch_size,e_dim)).cuda()
    users_preferences = []
    if dataset == 'explicit':
        pass
    if dataset == 'implicit':
        for i in range(0,2):
            zeros_preference = th.zeros((batch_size, e_dim), device='cuda')
            users_items = interaction_historys[i]  # list(batch_size,list(l))
            if len(users_items) == 0:
                users_preferences.append(zeros_preference)
                continue
            users_items_emb,seq_lens = pad_sequence(interaction_historys=interaction_historys, z=i, item_emb=item_emb,
                                           mask_emb=mask_emb)
            users_preference = ssab(users_items_emb, i)    # tensor(batch_szie,e_dim)
            for bs in range(batch_size):
                if seq_lens[bs] != 0:
                    zeros_preference[bs] = users_preference[bs]
            users_preferences.append(zeros_preference)
    users_preferences = th.cat(users_preferences,dim=1).cuda()               #tensor(batch_size,e_dim*6)
    return users_preferences

def update_user_preference(user_interaction_history,z,item_emb,ssab,user_preferences,item_gemb=None):
    """
    :param user_interaction_history: 当前评分的交互历史
    :param z: 当前评分
    :param item_emb: 所有item的embedding词典
    :param ssab: ssab实例
    :param user_preference: 用户的前一兴趣
    :return:
    """
    e_dim = item_emb.weight.shape[1]
    user_items = tensor(user_interaction_history,dtype=th.long).cuda()
    if (item_gemb == None):
        user_items_emb = item_emb(user_items)
    else:
        user_items_emb = item_gemb[user_items]
    user_preference = ssab(user_items_emb,z).squeeze(dim=0)
    user_preferences[z*e_dim:(z+1)*e_dim] = user_preference
    return user_preferences


def add_update_preference(user_interaction_history,z,item_emb,ddqn_pre,user_preferences):
    e_dim = item_emb.weight.shape[1]
    user_items = tensor(user_interaction_history, dtype=th.long).cuda()
    user_items_emb = item_emb(user_items)
    user_preference = th.mean(user_items_emb,dim=-2)
    user_preference = ddqn_pre(user_preference,z)
    user_preferences[z*e_dim:(z+1)*e_dim] = user_preference
    return user_preferences


def add_preference(user_interaction_history,item_emb,rnn):
    e_dim = item_emb.weight.shape[1]
    user_items = tensor(user_interaction_history, dtype=th.long).cuda()
    user_items_emb = item_emb(user_items)
    user_preference = th.mean(user_items_emb, dim=-2)
    return user_preference

def add_preferences(users_interaction_history,item_emb,mask_emb):
    users_items_emb = pad_sequence1(interaction_historys=users_interaction_history, item_emb=item_emb, mask_emb=mask_emb)
    user_preferences = th.mean(users_items_emb, dim=-2)
    return user_preferences




def add_users_preferences(users_interaction_history,item_emb,mask_emb,ddqn_pre,dataset='implicit'):
    if hasattr(th.cuda, 'empty_cache'):
        th.cuda.empty_cache()
    e_dim = item_emb.weight.shape[1]
    batch_size, interaction_historys,empty_flag = translist2dict(users_interaction_history=users_interaction_history,dataset=dataset)
    if empty_flag == True:
        return th.zeros((batch_size,e_dim)).cuda()
    users_preferences = []
    if dataset == 'explicit':
        pass
    if dataset == 'implicit':
        for i in range(0,2):
            zeros_preference = th.zeros((batch_size, e_dim), device='cuda')
            users_items = interaction_historys[i]  # list(batch_size,list(l))
            if len(users_items) == 0:
                users_preferences.append(zeros_preference)
                continue
            users_items_emb, seq_lens = pad_sequence(interaction_historys=interaction_historys, z=i, item_emb=item_emb,
                                           mask_emb=mask_emb)
            user_preference = th.mean(users_items_emb, dim=-2)
            users_preference = ddqn_pre(user_preference, i)
            users_preferences.append(users_preference)  # tensor(batch_szie,e_dim)
    users_preferences = th.cat(users_preferences,dim=1).cuda()               #tensor(batch_size,e_dim*6)
    return users_preferences




