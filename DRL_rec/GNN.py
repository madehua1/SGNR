# 构建一个2层的GNN模型
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import dgl.function as fn
from dgl.nn.pytorch import GATConv

class GCN(nn.Module):
    def __init__(self, in_feat=50, hid_feat=50, layers=2, bn_flag=False):
        super(GCN, self).__init__()
        self.gnns = nn.ModuleList()
        self.in_feat = in_feat
        self.hin_feat = hid_feat
        self.layers = layers
        self.bn_flag = bn_flag
        for i in range(layers):
            #self.gnn = dglnn.GraphConv(in_feat, hid_feat,norm="both", weight=True, bias=True)
            #self.gnn = dglnn.GATConv(in_feats=in_feat,out_feats=hid_feat,activation=F.relu,num_heads=5,feat_drop=0.5,attn_drop=0.5)
            #self.gnn = dglnn.GATConv(in_feats=in_feat,out_feats=hid_feat,activation=F.relu, num_heads=5)
            #self.gnn = dglnn.GATConv(in_feats=in_feat,out_feats=hid_feat,activation=F.relu,num_heads=1)
            self.gnn = dglnn.GATConv(in_feats=in_feat, out_feats=hid_feat,activation=F.relu, num_heads=1)
            self.gnn.reset_parameters()
            self.gnns.append(self.gnn)
            in_feat = hid_feat

    # def forward(self,graph,inputs):
    #     for idx, gnn in enumerate(self.gnns):
    #         if self.bn_flag:
    #             inputs = nn.BatchNorm1d(inputs.shape[1]).cuda()(inputs)
    #         hid = gnn(graph,inputs)
    #         hid = th.sum(hid,dim=-2)
    #         inputs = hid
    #     return inputs

    def block_forward(self,block,inputs,learning_flag=False):
        for idx, gnn in enumerate(self.gnns):
            # if self.bn_flag:
            #     if learning_flag:
            #         inputs = nn.BatchNorm1d(inputs.shape[1]).cuda()(inputs)
            hid = gnn(block[idx],inputs)
            hid = th.sum(hid,dim=-2)
            # if self.bn_flag:
            #     hid = nn.BatchNorm1d(hid.shape[1]).cuda()(hid)
            inputs = hid
        return inputs


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        #self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        if num_layers > 1:
        # input projection (no residual)
            self.gat_layers.append(GATConv(
                in_dim, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            # hidden layers
            for l in range(1, num_layers-1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers.append(GATConv(
                num_hidden * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))
        else:
            self.gat_layers.append(GATConv(
                in_dim, num_classes, heads[0],
                feat_drop, attn_drop, negative_slope, residual, None))

    def block_forward(self,blocks, inputs, learning_flag=False):
        h = inputs
        for l in range(self.num_layers):
            a = blocks[l]
            b = self.gat_layers[l]
            h = b(a,h)
            #h = self.gat_layers[l](blocks[l], h)
            h = h.flatten(1) if l != self.num_layers - 1 else h.mean(1)
        return h


class RNN(nn.Module):
    def __init__(self,in_feat=50,hid_feat=50,layers=2):
        super(RNN, self).__init__()
        self.rnns = nn.ModuleList()
        self.in_feat = in_feat
        for i in range(layers):
            self.rnn = nn.GRU(self.in_feat, self.in_feat, 1, batch_first=True).cuda()
            self.rnn.reset_parameters()
            self.rnns.append(self.rnn)
    def forward(self,inputs,z):
        """
        :param inputs: 输入张量     tensor(item_num,e_dim)
        :return
        """
        if th.is_tensor(inputs):
            if len(inputs.shape) == 2:
                inputs = inputs.unsqueeze(dim=0)
        out, h = self.rnns[z](inputs)
        return h[-1]



