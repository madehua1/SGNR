# --coding:utf-8--
import torch as th
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.Wq = nn.ModuleList()
        self.Wk = nn.ModuleList()
        self.Wv = nn.ModuleList()
        for i in range(0, 6):
            self.Wq.append(nn.Linear(in_features=dim, out_features=dim, bias=False))
            self.Wk.append(nn.Linear(in_features=dim, out_features=dim, bias=False))
            self.Wv.append(nn.Linear(in_features=dim, out_features=dim, bias=False))
        self.dim = dim
    def forward(self, Ez, z):
        '''
        需要对scores保留下三角
        :param Ez: 评分为z的item_emb    tensor(batch_size,lmax,e_dim)    补零后的embedding
        :param z: 评分z
        :return:
        '''
        assert Ez.shape[-1] == self.dim
        assert z >= 0 & z <= 5
        # 设置 `transpose_b=True` 为了交换 `keys` 的最后两个维度
        queries = self.Wq[z](Ez)
        keys = self.Wk[z](Ez)
        values = self.Wv[z](Ez)
        scores = th.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.dim)
        diag_vals = th.ones_like(scores[0, :, :])
        tril = th.tril(diag_vals)
        masks = tril.unsqueeze(0).repeat(scores.shape[0], 1, 1)
        paddings = th.ones_like(masks) * (-2 ** 32 + 1)
        scores = th.where(th.eq(masks, 0), paddings, scores)
        self.attention_weights = th.softmax(scores, dim=-1)
        a = th.matmul(self.attention_weights, values)
        return a

class feed_forward_layer(nn.Module):
    def __init__(self, dim):
        super(feed_forward_layer, self).__init__()
        self.fc1 = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.fc2 = nn.Linear(in_features=dim, out_features=dim, bias=True)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class stacking_SAB(nn.Module):
    """初始化时，输入SA模块，FF模块，SA和FF模块叠加层数"""
    def __init__(self, SA, FF, layers):
        super(stacking_SAB, self).__init__()
        self.layers = layers
        self.SA = SA
        self.FF = FF

    def forward(self, inputs,z):
        """
        :param inputs: 用户交互的item    tensor(batch_size, Lmax, e_dim)
        :param z: item交互的评分
        :return:
        """
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(dim=0)

        for layer in range(self.layers):
            Sz = self.SA(inputs, z)
            # Fz = []
            # for i in range(Sz.shape[1]):
            #     sz = Sz[:,i,:]
            #     fz = self.FF(sz)
            #     Fz.append(fz)
            Fz = [self.FF(Sz[:,i,:]) for i in range(Sz.shape[1])]

            Fz_tensor = th.stack(Fz,dim=1)
            inputs = Fz_tensor
        return th.sum(inputs,dim=1)

class add_pre(nn.Module):
    def __init__(self,e_dim):
        super(add_pre, self).__init__()
        self.e_dim = e_dim
        self.Lin = nn.ModuleList()
        self.Lin1 = nn.Linear(in_features=self.e_dim,out_features=self.e_dim,bias=True)
        self.Lin2 = nn.Linear(in_features=self.e_dim,out_features=self.e_dim,bias=True)
        self.Lin.append(self.Lin1)
        self.Lin.append(self.Lin2)
    def forward(self,input,z):
        #output = input
        output = self.Lin[z](input)
        return output
