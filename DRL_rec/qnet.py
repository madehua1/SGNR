import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self,candi_num, e_dim=50, hidden_dim=50,bn_flag=False,method='NISR',dataset='implicit',add_method='None'):
        """
        :param candi_num: 候选item个数
        :param e_dim: 嵌入向量的维数
        :param hidden_dim: 全连接层隐藏层的维数
        """
        super(QNet, self).__init__()
        self.bn_flag = bn_flag
        self.candi_num =candi_num
        self.fc1 = nn.Linear(e_dim, hidden_dim)
        if method == 'NICF':
            if dataset == 'explicit':
                self.fc1 = nn.Linear(e_dim*6, hidden_dim)
            if dataset == 'implicit':
                self.fc1 = nn.Linear(e_dim*2, hidden_dim)
                if add_method == 'concatenate':
                    self.fc1 = nn.Linear(e_dim * 4,hidden_dim)
        #V(s)
        self.fc2_value = nn.Linear(hidden_dim, hidden_dim)
        #self.fc3_value = nn.Linear(hidden_dim, hidden_dim)
        self.out_value = nn.Linear(hidden_dim, 1)
        #Q(s,a)
        self.fc2_advantage = nn.Linear(hidden_dim+e_dim, hidden_dim)   #hidden_dim + emb_size
        #self.fc3_advantage = nn.Linear(hidden_dim, hidden_dim)
        self.out_advantage = nn.Linear(hidden_dim,1)

        #BatchNorm
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(1)
        self.bn3 = nn.BatchNorm1d(self.candi_num)
        self.bn3_ = nn.BatchNorm1d(1)

    def forward(self, e_u, e_v, choose_action = True,learning_flag=False):
        """
        :param e_u:用户的状态向量    tensor([batch_size,e_dim * rating_num]) ;
        :param e_v:候选item的嵌入向量,由于每位用户的候选item可能不同，不能合并batch_size [batch_size,candi_items, e_dim]
        :return: qsa: 预测分数 [batch_size,candi_item_nums]
        """
        e_u = e_u.unsqueeze(dim=1)
        if self.bn_flag:
            x = F.relu(self.bn1(self.fc1(e_u)))
        else:
            x = F.relu(self.fc1(e_u))      #[batch_size, 1, hidden_dim]

        #v(s)

        #value = self.fc3_value(x)
        #value = F.relu(x)
        # value = self.fc2_value(x)
        # value = F.relu(value)
        # value = self.out_value(value)
        # value = value.squeeze(dim=1)

        if self.bn_flag:
            value = self.out_value(F.relu(self.bn2(self.fc2_value(x)))).squeeze(dim=1)
        else:
            value = self.out_value(F.relu(self.fc2_value(x))).squeeze(dim=1)  # [batch_size,1,1] -> [batch_size,1]       #squeeze维度压缩，删掉所有为1的维度

        #Q(s,a)
        if choose_action:
            x = x.repeat(1,self.candi_num,1)        #[batch_size,candi_item_nums,e_dim]
        state_cat_action = torch.cat((x, e_v), dim=2)  # [batch_size,candi_item_nums,e_dim*6+e_dim]

        # advantage = self.fc2_advantage(state_cat_action)
        # advantage = F.relu(advantage)
        # #advantage = self.fc3_advantage(advantage)
        # #advantage = F.relu(advantage)
        # advantage = self.out_advantage(advantage).squeeze(dim=2)


        if choose_action & self.bn_flag :
            advantage = self.out_advantage(F.relu(self.bn3(self.fc2_advantage(state_cat_action)))).squeeze(dim=2)
        elif self.bn_flag:
            advantage = self.out_advantage(F.relu(self.bn3_(self.fc2_advantage(state_cat_action)))).squeeze(dim=2)
        else:
            advantage = self.out_advantage(F.relu(self.fc2_advantage(state_cat_action))).squeeze(dim=2)                  #[batch_size*candi_item_nums]


        if choose_action:
            qsa = advantage + value - advantage.mean(dim=1, keepdim=True)
        else:
            qsa = advantage + value
        #qsa = advantage + value


        return qsa