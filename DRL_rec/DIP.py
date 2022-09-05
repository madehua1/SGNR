# --coding:utf-8--
import torch.nn as nn
import torch as th
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self,in_feat=50,hid_feat=50,layers=2,number=1):
        super(RNN, self).__init__()
        self.rnns = nn.ModuleList()
        self.in_feat = in_feat
        self.number = number
        for i in range(layers):
            self.rnn = nn.GRU(self.in_feat, self.in_feat, 1, batch_first=True).cuda()
            self.rnn.reset_parameters()
            self.rnns.append(self.rnn)

    def forward(self,inputs,z,h0=None):
        """
        :param inputs: 输入张量     tensor(item_num,e_dim)
        :return
        """
        if th.is_tensor(inputs):
            if len(inputs.shape) == 2:
                inputs = inputs.unsqueeze(dim=0)
        self.rnns[z].flatten_parameters()
        out, h = self.rnns[z](inputs)
        return h

class DIP(nn.Module):
    def __init__(self,RNNs):
        super(DIP, self).__init__()
        self.rnns = RNNs
    def forward(self,inputs,seq_lens_,z):
        _, idx_sort = th.sort(seq_lens_, dim=0, descending=True)
        _, idx_unsort = th.sort(idx_sort, dim=0)
        order_seq_lengths = th.index_select(seq_lens_, dim=0, index=idx_sort).cpu()
        order_item_emb_tensor = th.index_select(inputs, dim=0, index=idx_sort)
        x_packed = pack_padded_sequence(order_item_emb_tensor, order_seq_lengths, batch_first=True)
        h = self.rnns(x_packed,z)
        h = th.index_select(h[-1], dim=0, index=idx_unsort)
        users_preference = h.squeeze(dim=0)
        return users_preference

class attention(nn.Module):
    def __init__(self,e_dim):
        super(attention, self).__init__()
        self.fc1 = nn.Linear(in_features=e_dim*4,out_features=e_dim*2)
        self.fc2 = nn.Linear(in_features=e_dim*2,out_features=e_dim*2)
        self.bn1 = nn.BatchNorm1d(e_dim*2)
    def forward(self,inputs,method='NISR'):
        if method == 'NISR':
            outputs = self.fc1(inputs)
            # outputs = self.fc2(outputs)
            # outputs = F.relu(outputs)
            return outputs

class attention1(nn.Module):
    def __init__(self,e_dim):
        super(attention1, self).__init__()
        self.fc = nn.Linear(in_features=e_dim*2,out_features=e_dim*2)

    def forward(self,inputs1,inputs2):
        outputs = inputs1 + inputs2
        outputs = self.fc(outputs)
        return outputs

