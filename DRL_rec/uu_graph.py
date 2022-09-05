import numpy as np
import torch as th
import dgl
import torch.nn as nn
import torch.nn.functional as F
import math
device = "cuda" if th.cuda.is_available() else "cpu"

class U_Graph(nn.Module):
    def __init__(self, src, dst, e_dim,user_emb):
        super(U_Graph, self).__init__()
        self.src = src
        self.dst = dst
        self.e_dim = e_dim
        self.uu_graph_gen(user_emb)
    def uu_graph_gen(self,user_emb):
        user_num = user_emb.weight.shape[0]
        self.graph = dgl.graph((self.src, self.dst),num_nodes=user_num)
        # self.graph = dgl.to_bidirected(self.graph)
        self.graph = dgl.add_self_loop(self.graph)
        self.graph = self.graph.to(device)
        self.graph.ndata["feature"] = user_emb.weight














