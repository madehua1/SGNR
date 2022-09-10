# --coding:utf-8--
import torch
from collections import namedtuple
import torch.optim as optim
import random
import torch.nn as nn
import numpy as np
import torch as th
import DRL_rec.DRL_utils
import utils_
from DRL_rec.DRL_utils import get_users_perference,update_user_preference
import dgl
import math


Transition = namedtuple('Transition', ('user_id', 'interaction_history', 'action',
                                           'reward', 'next_interaction_history', 'next_candi','done_mask'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.BatchTransition = {}
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:  # 有容量时，增加元素，无容量时替换元素
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=1):
        return random.sample(self.memory, batch_size)

    def __len__(self):  # 被内置函数len()调用
        return len(self.memory)

class DQN(object):
    def __init__(self, n_action, learning_rate, eval_lr, gcn_lr, ssab_lr,rnn_lr, item_lr, user_lr, att_lr,add_lr, l2_norm,
                 graph, sampler, users, items_emb, user_emb, mask_emb, DIP, ssab,add, gcn_net, eval_net,
                 target_net, attention,attention1,memory_size,test_memory_size, eps_start, eps_end, eps_decay, al, norm,
                 batch_size, test_batch_size, gamma, target_update_step, Eta, max_step, update_times, tau=0.01, double_q=True,
                 attention_flag=True, preference_flag=True, social_flag=True,
                 method="NISR", dataset='implicit', pre_method='GRU'):
        """

        :param n_action:所有item的个数
        :param learning_rate: 学习率
        :param l2_norm: L2正则化
        :param items_emb: 商品的词典
        :param rnn:建模用户兴趣所使用的rnn
        :param gcn_net: 图卷积网络
        :param eval_net: 评估网络
        :param target_net: 目标网络
        :param memory_size:DQN的经验回放的大小
        :param eps_start:
        :param eps_end:
        :param eps_decay:
        :param batch_size:DQN更新的batch_size
        :param gamma:贝尔曼方程的中gamma
        :param target_update_step:
        :param tau: duel Q的更新tau
        :param double_q: 是否采样double_q
        """
        self.device = "cuda" if th.cuda.is_available() else "cpu"
        self.graph = graph
        self.sampler = sampler
        self.users = users
        self.items_emb = items_emb
        self.user_emb = user_emb
        self.mask_emb = mask_emb
        self.DIP = DIP
        self.ssab = ssab
        self.add = add
        self.gcn_net = gcn_net
        self.eval_net = eval_net
        self.target_net = target_net
        self.att = attention
        self.att1 = attention1
        self.memory = ReplayMemory(memory_size)
        self.test_memory_size = test_memory_size
        self.test_memory = ReplayMemory(self.test_memory_size)
        self.global_step = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.n_action = n_action
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.gamma = gamma
        self.eps_threshold = 0
        self.start_learning = 2000
        self.norm = norm
        self.target_update_step = target_update_step
        self.Eta = Eta
        self.max_step = max_step
        self.update_times = update_times
        self.tau = tau
        self.double_q = double_q
        self.attention_flag = attention_flag
        self.preference_flag = preference_flag
        self.social_flag = social_flag
        self.dataset = dataset
        self.pre_method = pre_method
        '''学习率'''
        self.lr = learning_rate
        self.eval_lr = eval_lr
        self.gcn_lr = gcn_lr
        self.ssab_lr = ssab_lr
        self.rnn_lr = rnn_lr
        self.item_lr = item_lr
        self.user_lr = user_lr
        self.att_lr = att_lr
        self.add_lr = add_lr
        self.al = al
        self.mul_rate = 2
        self.optimizer = optim.Adam([
            {"params": self.eval_net.parameters(), 'lr': self.eval_lr},
            {"params": self.gcn_net.parameters(), 'lr': self.gcn_lr},
            {"params": self.ssab.parameters(), 'lr': self.ssab_lr},
            {"params": self.DIP.rnns.parameters(), 'lr': self.rnn_lr},
            # {"params": self.add.parameters(), 'lr': self.add_lr},
            {"params": self.items_emb.parameters(), 'lr': self.item_lr},
            {"params": self.user_emb.parameters(), 'lr': self.user_lr},
            {"params": self.att.parameters(), 'lr': self.att_lr},
            {"params": self.att1.parameters(), 'lr': self.att_lr}
        ], weight_decay=l2_norm)
        # print('eval,item，gat,user,减小正则化')
        self.loss_func = nn.MSELoss()
        self.target_net.load_state_dict(self.eval_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False


    def get_user_pre(self,interaction_historys,z,item_emb,user_preferences):
        if self.pre_method == 'ssab':
            preference = DRL_rec.DRL_utils.update_user_preference(interaction_historys,z,item_emb,self.ssab,user_preferences)
        elif self.pre_method == 'rnn':
            preference = utils_.update_user_preference(interaction_historys, z, item_emb, self.DIP.rnns, user_preferences)
        elif self.pre_method == 'add':
            preference = DRL_rec.DRL_utils.add_update_preference(interaction_historys,z, item_emb, self.add, user_preferences)
        return preference

    def get_users_pre(self,users_interaction_history,item_emb,mask_emb,dataset):
        if self.pre_method == 'ssab':
            preference = DRL_rec.DRL_utils.get_users_perference(users_interaction_history,item_emb,mask_emb,self.ssab,dataset)
        elif self.pre_method == 'rnn':
            preference = utils_.get_users_perference(users_interaction_history,item_emb,mask_emb,self.DIP,dataset)
        elif self.pre_method == 'add':
            preference = DRL_rec.DRL_utils.add_users_preferences(users_interaction_history,item_emb,mask_emb,self.add,dataset)
        return preference


    def choose_action(self, interaction_historys,social_influence, candi,user_preferences, is_test=False):
        """
        :param interaction_history: 当前用户的交互记录            dict(6,list(l))
        :param social_influence: 当前用户的社交影响力             tensor(e_dim)
        :param candi：当前用户的候选item                          list(candi_num)
        :param is_test:是否为测试阶段
        :return:
        """
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        if (interaction_historys == None) & (self.social_flag != True):
            return random.choice(candi), user_preferences
        candi_tensor = th.LongTensor(candi).cuda()
        candi_emb = self.items_emb(candi_tensor)
        state_emb = user_preferences
        if (self.preference_flag) & (interaction_historys != None):
            z = interaction_historys[0]
            interaction_history = interaction_historys[1]
            state_emb = self.get_user_pre(interaction_historys=interaction_history,z=z,
                                               item_emb=self.items_emb,user_preferences=user_preferences)
        if self.social_flag & (not self.preference_flag):
            state_emb = social_influence
        elif self.social_flag & self.preference_flag & (not self.attention_flag):
            # state_emb = self.mul_rate * (self.al * state_emb + (1-self.al) * social_influence)
            state_emb = self.att1(state_emb,social_influence)
        elif self.social_flag & self.preference_flag & self.attention_flag:
            state_emb = th.cat([state_emb,social_influence],dim=-1)
            state_emb = self.att(state_emb)

        candi_emb = candi_emb.unsqueeze(dim=0)
        state_emb = th.unsqueeze(state_emb,dim=0)
        #self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.global_step * self.eps_decay)
        self.eps_threshold = 0.1
        # if self.global_step<10000:
        #     self.eps_threshold = 0.1

        # actions_value = self.eval_net(state_emb, candi_emb)
        # action = candi[actions_value.argmax().item()]
        if is_test or random.random() > self.eps_threshold:
            actions_value = self.eval_net(state_emb, candi_emb)
            action = candi[actions_value.argmax().item()]
        else:  # 进行随机选择，探索
            action = random.choice(candi)
        return action, user_preferences

    def learn(self,is_test=False):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        if len(self.memory) < self.start_learning:
            return

        if self.global_step % self.target_update_step == 0:
            for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
                target_param.data.copy_(param.data)
        # for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
        #     target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))



        #self.gamma = 1/(1+math.pow(max((self.max_step*epoch_model*self.update_times-self.global_step),0),self.Eta))
        self.global_step += 1
        #self.gamma = 0.5
        if is_test:
            transitions = self.test_memory.sample(self.test_batch_size)
        else:
            transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        batch_user_id = list(batch.user_id)
        batch_dm = th.tensor(list(batch.done_mask)).unsqueeze(dim=1).cuda()
        b_s_e = th.zeros((self.batch_size,self.items_emb.weight.shape[1]*2),device='cuda',dtype=th.float)
        b_s_e_ = th.zeros((self.batch_size,self.items_emb.weight.shape[1]*2),device='cuda',dtype=th.float)
        node_dataloader = dgl.dataloading.NodeDataLoader(self.graph.graph,self.users, self.sampler,
                                                         device=self.device,
                                                         batch_size=len(self.users), shuffle=False,
                                                         drop_last=False,
                                                         num_workers=0)
        for (inputs_nodes, seeds, blocks) in node_dataloader:
            inputs = self.user_emb(inputs_nodes)
            social_influences = self.gcn_net.block_forward(blocks, inputs)
            social_influences = social_influences[batch_user_id]
        if self.preference_flag:
            b_s_e = self.get_users_pre(batch.interaction_history,self.items_emb,self.mask_emb,dataset=self.dataset)
            b_s_e_ = self.get_users_pre(batch.next_interaction_history,self.items_emb,self.mask_emb,dataset=self.dataset)
        if self.social_flag & (not self.preference_flag):
            b_s_e = social_influences
            b_s_e_ = social_influences
        elif self.social_flag & self.preference_flag & (not self.attention_flag) :
            # b_s_e = self.mul_rate * (self.al * b_s_e +  (1-self.al) * social_influences)
            b_s_e = self.att1(b_s_e,social_influences)
            # b_s_e_ = self.mul_rate * (self.al * b_s_e_ + (1-self.al) * social_influences)
            b_s_e_ = self.att1(b_s_e_,social_influences)
        elif self.social_flag & self.attention_flag & self.preference_flag:
            b_s_e = th.cat([b_s_e, social_influences], dim=-1)
            b_s_e_ = th.cat([b_s_e_,social_influences],dim=-1)
            b_s_e = self.att(b_s_e)
            b_s_e_ = self.att(b_s_e_)
        b_a = torch.tensor(np.array(batch.action).reshape(-1, 1),dtype=th.long,device='cuda')  # [N*1]
        # b_a = torch.tensor(list(batch.action),dtype=th.long,device='cuda')  # [N*1]
        b_a_emb = self.items_emb(b_a)                                         # [N*1*emb_dim]
        b_r = torch.FloatTensor(np.array(batch.reward).reshape(-1, 1)).cuda()
        next_candi = torch.LongTensor(list(batch.next_candi)).cuda()
        next_candi_emb = self.items_emb(next_candi)  # [N*k*emb_dim]

        q_eval = self.eval_net(b_s_e, b_a_emb, choose_action=False ,learning_flag=True)
        if is_test:
            best_actions = torch.gather(input=next_candi, dim=1,
                                    index=self.eval_net(b_s_e_, next_candi_emb).argmax(dim=1).view(self.test_batch_size,
                                                                                                   1).cuda())
        else:
            best_actions = torch.gather(input=next_candi, dim=1,
                                    index=self.eval_net(b_s_e_, next_candi_emb).argmax(dim=1).view(self.batch_size,
                                                                                                   1).cuda())
        best_actions_emb = self.items_emb(best_actions)
        if self.double_q:
            q_target = b_r + self.gamma * batch_dm * self.target_net(b_s_e_, best_actions_emb,
                                                                   choose_action=False,learning_flag=True).detach()
        else:
            q_target = b_r + self.gamma * batch_dm * self.eval_net(b_s_e_, best_actions_emb, choose_action=False).detach()
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()  # 所有参数的梯度设置为0
        loss.backward()  # 得到所有参数的梯度
        self.optimizer.step()  # 更新所有的参数
