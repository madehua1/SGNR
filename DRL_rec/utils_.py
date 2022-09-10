import os
import pickle
import sys

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn, tensor
from torch.nn.utils.rnn import pack_padded_sequence
from DRL_rec.DRL_utils import translist2dict



p = os.path.dirname(os.path.dirname((os.path.abspath('__file__'))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append(r'../数据集douban')

def pickle_save(object, file_path):
    """
    完成对象的封装
    """
    f = open(file_path, 'wb')
    pickle.dump(object, f)

def pickle_load(file_path):
    """
    对象的读取
    """
    f = open(file_path, 'rb')
    return pickle.load(f)


def get_data(object_path, social_path):
    """
    :param object_path: 交互数据集路径
    :param social_path: 社交关系数据集路径
    """
    object = pickle_load(object_path)
    rating_mat = object["rating_mat"]
    user_num = object["user_num"]
    item_num = object["item_num"]
    rela_num = object["rela_num"]
    social = np.loadtxt(social_path)
    u_src, u_dst = [], []
    for u in social:
        u_src.append(int(u[0]))
        u_dst.append(int(u[1]))
    return rating_mat, u_src, u_dst

def get_data_(object_path, similarity_path):
    object = pickle_load(object_path)
    rating_mat = object["rating_mat"]
    user_num = object["user_num"]
    item_num = object["item_num"]
    rela_num = object["rela_num"]

    similarity = np.loadtxt(similarity_path)
    i_src, i_dst = [], []
    for s in similarity:
        i_src.append(int(s[0]))
        i_dst.append(int(s[1]))
    return rating_mat, i_src, i_dst


def get_item_emb(MF_model_path):
    MF_model = pickle_load(MF_model_path)
    item_emb = MF_model.embed_item.weight
    user_emb = MF_model.embed_user.weight
    item_emb = nn.Embedding(item_emb.shape[0], item_emb.shape[1], max_norm=1).from_pretrained(item_emb,
                                                                                              freeze=False).cuda()
    #item_emb = nn.Embedding(item_emb.shape[0], item_emb.shape[1], max_norm=1).cuda()
    user_emb = nn.Embedding(user_emb.shape[0], user_emb.shape[1], max_norm=1).from_pretrained(user_emb,
                                                                                              freeze=False).cuda()
    return item_emb, user_emb


def get_item_emb_user(item_path, user_path, e_dim,method=None,feature_path=None):
    items = pickle_load(item_path)
    users = pickle_load(user_path)
    item_emb = nn.Embedding(num_embeddings=len(items), embedding_dim=e_dim).cuda()
    user_emb = nn.Embedding(num_embeddings=len(users), embedding_dim=e_dim * 2).cuda()
    if method=='SADQN':
        user_emb = nn.Embedding(num_embeddings=len(users),embedding_dim=e_dim).cuda()
    nn.init.normal_(user_emb.weight, mean=0, std=0.01)
    nn.init.normal_(item_emb.weight, mean=0, std=0.01)
    if feature_path != None:
        feature = np.loadtxt(fname=feature_path,dtype=float,delimiter='\t')
        feature = th.tensor(feature,dtype=th.float32)
        item_emb = nn.Embedding(num_embeddings=len(items), embedding_dim=e_dim).from_pretrained(feature,freeze=True).cuda()
    return item_emb, user_emb


"""根据用户选择的item，使用rnn得到用户的动态兴趣"""


def user_preference(user_items, itempool_emb, rnn):  # user_items      list(l)   l为当前用户选择的item个数
    items_emb = itempool_emb(th.tensor(user_items).cuda())  # tensor(item_num, e_dim)
    h = rnn(items_emb)
    h = h.squeeze(dim=0).squeeze(dim=0)
    # tensor(e_dim)
    user_preference = h
    return user_preference


def users_preference(users_items, item_emb, mask_emb, rnn):  # users_items        list(batch_size,l)
    if hasattr(th.cuda, 'empty_cache'):
        th.cuda.empty_cache()
    batch_size = len(users_items)
    seq_lens = [len(list) for list in users_items]
    max_len = max(seq_lens)
    zeros_preference = th.zeros(batch_size, item_emb.weight.shape[1]).cuda()

    """batch中所有用户交互的item数量均为0，返回空的偏好向量"""
    if max_len == 0:
        return zeros_preference

    """计算batch中交互item非空的用户的偏好向量"""
    users_items = [list for list in users_items if len(list) > 0]
    seq_lens_ = [len(list) for list in users_items]
    user_items = [th.tensor(list) for list in users_items]
    user_items_pad = th.nn.utils.rnn.pad_sequence(user_items, batch_first=True, padding_value=0).to(th.long).cuda()
    item_emb_tensor = item_emb(user_items_pad)
    zeros = th.zeros_like(item_emb_tensor)
    mask = mask_emb(user_items_pad)
    item_emb_tensor = th.where(th.eq(mask, 0), zeros, item_emb_tensor)

    """使用RNN处理变长序列"""
    seq_lens_ = th.IntTensor(seq_lens_).cuda()
    _, idx_sort = th.sort(seq_lens_, dim=0, descending=True)
    _, idx_unsort = th.sort(idx_sort, dim=0)
    order_seq_lengths = th.index_select(seq_lens_, dim=0, index=idx_sort).cpu()
    order_item_emb_tensor = th.index_select(item_emb_tensor, dim=0, index=idx_sort)
    x_packed = pack_padded_sequence(order_item_emb_tensor, order_seq_lengths, batch_first=True)
    h = rnn(x_packed)
    h = th.index_select(h[-1], dim=0, index=idx_unsort)
    users_preference = h.squeeze(dim=0)

    """batch用户偏好向量组合"""
    pre_ = 0
    for bs in range(batch_size):
        if seq_lens[bs] != 0:
            zeros_preference[bs] = users_preference[pre_]
            pre_ += 1
    users_preference = zeros_preference
    return users_preference  # tensor(batch_size,e_dim)


def update_user_preference(user_interaction_history, z, item_emb, rnns, user_preferences,item_gemb=None):
    """
    :param user_interaction_history: 当前评分的交互历史
    :param z: 当前评分
    :param item_emb: 所有item的embedding词典
    :param GRU: ssab实例
    :param user_preference: 用户的前一兴趣
    :return:
    """
    e_dim = item_emb.weight.shape[1]
    user_items = tensor(user_interaction_history, dtype=th.long).cuda()
    if (item_gemb == None):
        user_items_emb = item_emb(user_items)
    else:
        user_items_emb = item_gemb[user_items]
    user_preference = rnns(user_items_emb, z).squeeze(dim=0)
    user_preferences[z * e_dim:(z + 1) * e_dim] = user_preference
    return user_preferences


def pad_sequence(interaction_historys,z,item_emb,mask_emb,item_gemb=None):
    """将list(batch_size,tensor(l,e_dim))补齐，其中l不相等"""
    users_items = interaction_historys[z]  # list(batch_size,list(l))
    item_num = item_emb.weight.shape[0]
    seq_lens = [len(list) for list in users_items]
    users_items = [list for list in users_items if len(list) > 0]
    seq_lens_ = [len(list) for list in users_items]
    seq_lens_ = th.IntTensor(seq_lens_).cuda()
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
    return users_items_emb, seq_lens, seq_lens_

def get_users_perference(users_interaction_history, item_emb, mask_emb, DIP, dataset='explicit',item_gemb=None):
    """
    :param users_interation_history: batch用户的交互历史         list(batch_size*dict(6,list(l)))
    :param item_emb:所有item的embedding字典
    :param GRU:ssab实例
    :return:
    """
    item_num, e_dim = item_emb.weight.shape[0] ,item_emb.weight.shape[1]
    batch_size, interaction_historys, empty_flag = translist2dict(users_interaction_history=users_interaction_history,
                                                                  dataset=dataset)
    if empty_flag == True:
        return th.zeros((batch_size, e_dim)).cuda()
    users_preferences = []
    if dataset == 'explicit':
        pass
    if dataset == 'implicit':
        for i in range(0, 2):
            zeros_preference = th.zeros((batch_size, e_dim), device='cuda')
            users_items = interaction_historys[i]  # list(batch_size,list(l))
            users_items = [list for list in users_items if len(list) > 0]
            if len(users_items) == 0:
                users_preferences.append(zeros_preference)
                continue
            users_items_emb, seq_lens, seq_lens_ = pad_sequence(interaction_historys, z=i, item_emb=item_emb,mask_emb=mask_emb,item_gemb=item_gemb)
            users_preference = DIP(users_items_emb, seq_lens_, i)
            """batch用户偏好向量组合"""
            pre_ = 0
            for bs in range(batch_size):
                if seq_lens[bs] != 0:
                    zeros_preference[bs] = users_preference[pre_]
                    pre_ += 1
            users_preferences.append(zeros_preference)
    users_preferences = th.cat(users_preferences, dim=1).cuda()  # tensor(batch_size,e_dim*6)
    return users_preferences


def update_user_preference_(user_interaction_history, z, item_emb, rnns,users_preferences,item_gemb=None):
    """
    :param user_interaction_history: 当前评分的交互历史
    :param z: 当前评分
    :param item_emb: 所有item的embedding词典
    :param GRU: ssab实例
    :param user_preference: 用户的前一兴趣
    :return:
    """
    e_dim = item_emb.weight.shape[1]
    user_items = tensor(user_interaction_history, dtype=th.long).cuda()
    user_items_emb = item_emb(user_items)
    if item_gemb != None :
        user_items_emb = item_gemb[user_items]
    if z == 1:
        users_preferences = rnns(user_items_emb, z).squeeze(dim=0)
    return users_preferences

def get_users_perference_(users_interaction_history, item_emb, mask_emb, DIP, dataset='explicit',item_gemb=None):
    """
    :param users_interation_history: batch用户的交互历史         list(batch_size*dict(6,list(l)))
    :param item_emb:所有item的embedding字典
    :param GRU:ssab实例
    :return:
    """
    item_num, e_dim = item_emb.weight.shape[0] ,item_emb.weight.shape[1]
    batch_size, interaction_historys, empty_flag = translist2dict(users_interaction_history=users_interaction_history,
                                                                  dataset=dataset)
    if empty_flag == True:
        return th.zeros((batch_size, e_dim)).cuda()
    users_preferences = []
    if dataset == 'implicit':
        zeros_preference = th.zeros((batch_size, e_dim), device='cuda')
        users_items = interaction_historys[1]  # list(batch_size,list(l))
        users_items = [list for list in users_items if len(list) > 0]
        if len(users_items) == 0:
            users_preferences = zeros_preference
        else:
            users_items_emb, seq_lens, seq_lens_ = pad_sequence(interaction_historys, z=1, item_emb=item_emb,mask_emb=mask_emb,item_gemb=item_gemb)
            users_preference = DIP(users_items_emb, seq_lens_, 1)
            """batch用户偏好向量组合"""
            pre_ = 0
            for bs in range(batch_size):
                if seq_lens[bs] != 0:
                    zeros_preference[bs] = users_preference[pre_]
                    pre_ += 1
            users_preferences = zeros_preference
    return users_preferences




def vectorize(M):
    '''
    对矩阵进行向量化
    :param M: 要向量化的矩阵
    :return: 向量化后得到的向量
    '''
    return np.reshape(M.T, M.shape[0] * M.shape[1])


def matrixize(V, C_dimension):
    '''
    对向量进行矩阵化
    :param V: 要矩阵化的向量
    :param C_dimension: 矩阵的列数
    :return: 矩阵化后的向量
    '''
    return np.transpose(np.reshape(V, (int(len(V) / C_dimension), C_dimension)))




