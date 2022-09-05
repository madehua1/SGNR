# --coding:utf-8--
import copy
import math
import logging
import os
import pickle

import dgl
import torch
import random
import numpy as np
from tqdm import tqdm

from DIP import attention, DIP, RNN,attention1
from GNN import GCN
from env import Env
from DRL_rec.NICF_dqn import DQN
from qnet import QNet
import time
import utils
import torch as th
import torch.nn as nn
from dataset import Userdata
from DRL_rec.Stacking_SAB import SelfAttention, feed_forward_layer, stacking_SAB, add_pre
from torch.utils.data import DataLoader
import torch.nn.functional as F
from uu_graph import U_Graph


def set_global_seeds(i):
    torch.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)
    torch.backends.cudnn.deterministic = True

class Recommender(object):
    def __init__(self, args):

        set_global_seeds(0)
        self.method = args.method
        self.dataset_name = args.dataset_name
        self.device = "cuda" if th.cuda.is_available() else "cpu"

        # dataset
        self.item_emb, self.user_emb = utils.get_item_emb_user(args.item_path, args.user_path, args.e_dim)
        # self.item_emb_, _ = utils.get_item_emb(args.MF_path)
        # self.item_emb_.weight.data = F.normalize(self.item_emb_.weight.data,p=2,dim=1)
        self.user_num = self.user_emb.weight.shape[0]
        # self.items_emb, user_num = utils.get_item_emb(args.MF_model_path)
        self.e_dim = self.item_emb.weight.shape[1]
        e_dim = self.e_dim
        self.item_num = self.item_emb.weight.shape[0]
        self.mask_emb = th.nn.Embedding(num_embeddings=self.item_num, embedding_dim=self.e_dim).cuda()
        self.mask_emb.weight[0] = th.zeros(self.e_dim)
        self.sample_size = args.sample_size
        self.test_sample_size = args.test_sample_size
        self.alpha = args.alpha
        self.dataset = args.dataset
        self.max_rating = args.max_rating
        self.min_rating = args.min_rating
        self.boundary_rating = args.boundary_rating
        self.episode_length = args.episode_length
        self.test_rec_round = args.test_rec_round
        self.test_rec_round_list = [5, 10, 20]
        rating_mat, u_src, u_dst = utils.get_data(args.object_path, args.social_path)
        # self.delta = [random.random() for i in range(self.user_num)]
        self.delta = [0.5 for i in range(self.user_num)]
        self.env = Env(episode_length=self.episode_length, alpha=self.alpha, boundary_rating=self.boundary_rating,
                       max_rating=self.max_rating, min_rating=self.min_rating, env_path=args.object_path, method='NICF',
                       dataset=self.dataset,delta=self.delta)
        # 用户
        u_src = th.tensor(u_src).cuda()
        u_dst = th.tensor(u_dst).cuda()
        users = np.array([i for i in range(self.user_num)], dtype=np.int64)
        items = np.array([i for i in range(self.item_num)],dtype=np.int64)
        self.sort = args.sort
        self.users_ = Userdata(users)

        if self.dataset_name == 'LastFM':
            number = 220
        else:
            number = 1020
        is_break = False
        for i in range(self.user_num):
            positive_num = self.read_user_im(i)
            if len(positive_num) >= number:
                train_num = i
                is_break = True
                break
        if not is_break:
             train_num = self.user_num - 1
        #train_num=1
        users_less = [i for i in range(train_num)]
        users_more = np.array([i+train_num for i in range(self.user_num - train_num)], dtype=np.int64)
        if not self.sort:
            np.random.shuffle(users_more)
        self.user_test_num = int(self.user_num * args.test_rate)
        self.user_train_num = self.user_num - self.user_test_num
        user_test = users_more[(-self.user_test_num):]
        user_train = list(set(users_less).union(set(users_more) - set(user_test)))
        # user_test = users_more[-self.user_test_num:-self.user_test_num+300]
        self.user_train = Userdata(th.tensor(np.array(user_train)))
        self.user_test = Userdata(th.tensor(user_test))



        self.user_train_dataloader = DataLoader(dataset=self.user_train,batch_size=self.sample_size,shuffle=True,num_workers=0)
        # Env
        self.dataset = args.dataset
        self.max_rating = args.max_rating
        self.min_rating = args.min_rating
        self.boundary_rating = args.boundary_rating
        self.episode_length = args.episode_length
        self.test_rec_round = args.test_rec_round
        self.delta = [0.7 for i in range(self.user_num)]
        self.env = Env(episode_length=self.episode_length, alpha=self.alpha, boundary_rating=self.boundary_rating,
                       max_rating=self.max_rating, min_rating=self.min_rating, env_path=args.object_path, method='NICF',
                       dataset=self.dataset,delta=self.delta)
        self.user_num, self.item_num, self.rela_num = self.env.get_init_data()
        self.train_rate = args.train_rate


        # DQN
        self.gamma = args.gamma
        self.Eta = args.Eta
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.memory_size = args.memory_size
        self.test_memory_size = args.test_memory_size
        self.bn_flag = args.bn_flag
        self.candi_num = args.candi_num
        self.fix_emb = args.fix_emb

        # train
        self.test_num = 0
        self.train_num = 0
        self.max_training_step = args.max_step
        self.ver_step = args.ver_step
        self.target_update_step = args.update_step
        self.update_times = args.update_times
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.double_q = args.double_q
        self.preference_flag = args.preference_flag
        self.social_flag = args.social_flag
        self.attention_flag = args.attention_flag
        self.norm = args.norm
        self.tau = args.tau
        self.pre_method = args.pre_method
        self.learning_rate = args.lr
        self.eval_lr = args.eval_lr
        self.gcn_lr = args.gcn_lr
        self.rnn_lr = args.rnn_lr
        self.ssab_lr = args.ssab_lr
        self.item_lr = args.item_lr
        self.user_lr = args.user_lr
        self.att_lr = args.att_lr
        self.add_lr = args.add_lr
        self.al = args.al
        self.l2_norm = args.l2
        self.load = args.load
        self.gcn_layer = args.gcn_layer
        self.model_name = args.modelname
        self.load_path = args.model_path + 'model_v%ds%s' % (args.version,args.modelname)
        self.save_step = args.save_step
        self.version = args.version
        self.save_path = args.model_path + 'model_v%d' % (args.version)

        #需要使用到的网络结构
        self.SA = SelfAttention(dim=self.e_dim)
        self.FF = feed_forward_layer(dim=self.e_dim)
        self.ssab = stacking_SAB(SA=self.SA, FF=self.FF, layers=1).cuda()
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.gcn_layer)
        self.graph = U_Graph(u_src, u_dst, self.e_dim, self.user_emb).cuda()
        self.eval_net = QNet(candi_num=self.candi_num, e_dim=self.e_dim, hidden_dim=self.e_dim*3, bn_flag=self.bn_flag,
                             method='NICF', dataset=self.dataset).cuda()
        self.target_net = QNet(candi_num=self.candi_num, e_dim=e_dim, hidden_dim=self.e_dim*3, bn_flag=self.bn_flag,
                               method='NICF', dataset=self.dataset).cuda()
        self.gcn = GCN(in_feat=e_dim * 2, hid_feat=e_dim * 2, layers=self.gcn_layer, bn_flag=self.bn_flag).cuda()
        self.rnns = RNN(in_feat=e_dim, hid_feat=e_dim).cuda()
        self.DIP = DIP(self.rnns)
        self.add = add_pre(e_dim=e_dim).cuda()
        self.att = attention(e_dim=e_dim).cuda()
        self.att1 = attention1(e_dim=e_dim).cuda()
        self.start_epoch = 0
        self.count = 0
        self.learn_time = 0
        if self.load:
            self.load_model()
        self.dqn = DQN(n_action=self.item_num, learning_rate=self.learning_rate, eval_lr=self.eval_lr,
                       gcn_lr=self.gcn_lr, ssab_lr=self.ssab_lr,rnn_lr=self.rnn_lr, item_lr=self.item_lr, user_lr=self.user_lr,
                       att_lr=self.att_lr, add_lr=self.add_lr, l2_norm=self.l2_norm, graph=self.graph, sampler=self.sampler,
                       users=self.users_, items_emb=self.item_emb, user_emb=self.user_emb,
                       mask_emb=self.mask_emb, DIP=self.DIP, ssab=self.ssab, add=self.add, gcn_net=self.gcn, eval_net=self.eval_net,
                       target_net=self.target_net, attention=self.att,attention1 = self.att1,
                       memory_size=self.memory_size,test_memory_size=self.test_memory_size, eps_start=self.eps_start, eps_end=self.eps_end,
                       eps_decay=self.eps_decay, al=self.al, norm=self.norm,
                       batch_size=self.batch_size,test_batch_size=self.test_batch_size, gamma=self.gamma, target_update_step=self.target_update_step,
                       Eta=self.Eta, max_step=self.max_training_step, update_times=self.update_times, tau=self.tau,
                       double_q=self.double_q, attention_flag=self.attention_flag, social_flag=self.social_flag,
                       preference_flag=self.preference_flag, dataset=self.dataset, pre_method=self.pre_method)
        if self.load:
            self.load_opt()

        self.items = set([i for i in range(self.item_num)])
        self.result_file_path = args.result_path + time.strftime(
            '%Y%m%d%H%M%S') + '_' + self.dataset + '_%f' % self.alpha
        self.storage = []


    def read_user_ex(self, user_id):             #douban
        col_index = self.env.rating_mat[user_id].nonzero()
        item_s = set(col_index[0].tolist())
        return item_s
    def candidate_ex(self,item_s,mask):        #douban
        item_s_remain = item_s - set(mask)
        if len(item_s_remain) >= self.candi_num:
            candi = random.sample(item_s_remain,self.candi_num)
            candi = list(candi)
        else:
            tmp = set(self.items) - item_s -set(mask)
            candi1 = list(random.sample(tmp,self.candi_num-len(item_s_remain)))
            candi2 = list(item_s_remain)
            candi = candi1 + candi2
        assert len(candi) == self.candi_num
        # candi = th.LongTensor(candi)
        candi = np.array(candi,dtype=np.longlong)
        candi = list(candi)
        return candi

    def read_user_im(self,user_id):
        col_index = self.env.rating_mat[user_id].nonzero()
        item_s = set(col_index[0].tolist())
        return item_s


    def candidate_im(self,item_s,mask):        #LastFM
        item_s_remain = item_s - set(mask)
        candi = random.sample(item_s_remain,self.candi_num)
        return candi

    def candi_generate(self,item_s,mask):
        if self.dataset == 'explicit':
            candi = self.candidate_ex(item_s, mask)
        if self.dataset == 'implicit':
            candi = self.candidate_im(item_s, mask)
        return candi

    def candi_gen(self,mask):
        candi = set(self.items) - set(mask)
        candi = random.sample(candi,self.candi_num)
        return candi


    def train(self):
        logs_list = []
        users_list = []
        for inx,users in tqdm(enumerate(self.user_train_dataloader),total=len(self.user_train_dataloader),ascii=True,desc="训练",ncols=50):
            users_ = users.cuda()
            node_dataloader = dgl.dataloading.NodeDataLoader(self.graph.graph, self.users_, self.sampler,
                                                             device=self.device,
                                                             batch_size=len(self.users_), shuffle=False,
                                                             drop_last=False,
                                                             num_workers=0)
            for _, (input_nodes, seeds, blocks) in enumerate(node_dataloader):
                social_influence = self.dqn.gcn_net.block_forward(blocks, self.dqn.user_emb(input_nodes))
            for idx, user in enumerate(users):
                user_id = user.item()
                users_list.append(user_id)
                state, cur_interaction_history = self.env.reset(user_id)
                cumul_reward, done = 0, False
                mask = []
                step = 0
                if self.dataset == 'explicit':
                    user_perferences = th.zeros(self.e_dim*6).cuda()
                    item_s = self.read_user_ex(user_id)
                    candi = self.candidate_ex(item_s, mask)
                if self.dataset == 'implicit':
                    user_perferences = th.zeros(self.e_dim * 2).cuda()
                    item_s = self.read_user_im(user_id)
                    #item_s = set(self.items)
                    candi = self.candidate_im(item_s, mask)
                    #candi = self.candi_gen(mask)
                interaction_historys = None
                while not done:
                    action_chosen, user_perferences = self.dqn.choose_action(interaction_historys=interaction_historys,
                                                                             social_influence=social_influence[user_id], candi=candi,
                                                                                user_preferences=user_perferences)
                    state, next_interaction_history, r, reward, done = self.env.step(action_chosen)
                    interaction_historys = [r, next_interaction_history[r]]
                    mask.append(action_chosen)
                    candi = self.candidate_im(item_s,mask)
                    #candi = self.candi_gen(mask)
                    done_mask = 1
                    if done:
                        done_mask = 0
                    lens = len(cur_interaction_history[0]) + len(cur_interaction_history[1])
                    if not ((self.social_flag == False) & (lens==0)):
                        self.dqn.memory.push(user_id, cur_interaction_history, action_chosen, reward, next_interaction_history, candi, done_mask)
                    cur_interaction_history = next_interaction_history
                    cumul_reward += reward
                    step += 1
                    log_list = [user_id, step, len(state), cumul_reward]
                    logs_list.append(log_list)

            for itr in range(self.update_times):
                self.dqn.learn()
            self.learn_time+=1
            self.count += 1
        self.save_model(self.model_name)
        self.model_name += 1
        metric_lists = self.get_metric(logs_list=logs_list,users=users_list,rec_round=self.episode_length)
        print(metric_lists)
        return metric_lists

    def evaluate(self,is_test=True):
        logs_list = []
        ave_reward = []
        tp_list = []
        users_list = []
        node_dataloader = dgl.dataloading.NodeDataLoader(self.graph.graph, self.users_, self.sampler,
                                                         device=self.device,
                                                         batch_size=len(self.users_), shuffle=False,
                                                         drop_last=False,
                                                         num_workers=0)
        for _, (input_nodes, seeds, blocks) in enumerate(node_dataloader):
            social_influence = self.dqn.gcn_net.block_forward(blocks, self.dqn.user_emb(input_nodes))
        for idx, user_id in tqdm(enumerate(self.user_test), total=len(self.user_test), ascii=True,desc="评估",ncols=50):
            user_id = int(user_id)
            users_list.append(user_id)
            cumul_reward, done = 0, False
            state, cur_interaction_history = self.env.reset(user_id)
            step = 0
            mask = []
            if self.dataset == 'explicit':
                user_perferences = th.zeros(self.e_dim * 6).cuda()
                item_s = self.read_user_ex(user_id)
                candi = self.candidate_ex(item_s, mask)
            if self.dataset == 'implicit':
                user_perferences = th.zeros(self.e_dim * 2).cuda()
                item_s = self.read_user_im(user_id)
                #item_s = set(self.items)
                candi = self.candidate_im(item_s, mask)
                #candi = self.candi_gen(mask)
            interaction_historys = None
            while not done:
                action_chosen, user_perferences = self.dqn.choose_action(interaction_historys=interaction_historys,
                                                                         social_influence=social_influence[user_id], candi=candi,
                                                                         user_preferences=user_perferences, is_test=True)
                state, next_interaction_history, r, reward, done = self.env.step(action_chosen,test_rec_round=self.test_rec_round)
                if r==None:
                    interaction_historys = None
                else:
                    interaction_historys = [r, next_interaction_history[r]]
                mask.append(action_chosen)
                candi = self.candidate_im(item_s,mask)
                #candi = self.candi_gen(mask)
                done_mask = 1
                if done:
                    done_mask = 0
                self.dqn.test_memory.push(user_id, cur_interaction_history, action_chosen, reward, next_interaction_history, candi, done_mask)
                cur_interaction_history = next_interaction_history
                cumul_reward += reward
                step += 1
                log_list = [user_id, step, len(state), cumul_reward]
                logs_list.append(log_list)
            ave = float(cumul_reward) / float(step)
            tp = float(len(state))
            ave_reward.append(ave)
            tp_list.append(tp)
        test_ave_reward = np.mean(np.array(ave_reward))

        precision = np.array(tp_list) / self.test_rec_round
        recall = np.array(tp_list) / (self.rela_num[self.user_test.users.numpy()] + 1e-20)

        # f1 = (2 * precision * recall) / (precision + recall + 1e-20)
        train_ave_precision = np.mean(precision[:self.user_train_num])
        train_ave_recall = np.mean(recall[:self.user_train_num])
        # train_ave_f1 = np.mean(f1[:self.user_train_num])
        test_ave_precision = np.mean(precision)
        test_ave_recall = np.mean(recall)
        # test_ave_f1 = np.mean(f1[self.user_train_num:self.user_num])

        metric_lists = self.get_metric(logs_list=logs_list,users=users_list,rec_round=self.test_rec_round)
        # utils.pickle_save(self.storage, self.result_file_path)

        print('\ttest  average reward over step: %2.4f, precision@%d: %.4f, recall@%d: %.4f' % (
            test_ave_reward, self.episode_length, test_ave_precision, self.episode_length, test_ave_recall))
        print(metric_lists)
        return metric_lists



    def test_(self,stage='train', metric_lists=None,test_rec_round=20):
        path = r'数据集%s/result/%s/%sv%dresults%d.pkl'%(self.dataset_name,self.method,stage,self.version,test_rec_round)
        if os.path.exists(path):
            results = pickle.load(open(path, 'rb'))
        else:
            results = []
        if (self.train_num == 0) & (stage=='train'):
            results = []
            self.train_num += 1
        if (self.test_num == 0) & (stage == 'test'):
            results = []
            self.test_num += 1
        results.append(metric_lists)
        pickle.dump(results, open(path, 'wb'))


    def get_metric(self, logs_list,users,rec_round):
        logs_array = np.array(logs_list)
        user_change_loc = np.linspace(0, len(logs_list), int(len(logs_list)/rec_round),endpoint=False).astype(np.int)
        metric_lists = []
        for test_rec_round in self.test_rec_round_list:
            step_ = logs_array[user_change_loc + test_rec_round-1, 1].astype(np.float64)
            tp_ = logs_array[user_change_loc + test_rec_round-1, 2].astype(np.float64)
            cumul_reward_ = logs_array[user_change_loc + test_rec_round-1, 3].astype(np.float64)
            ave_reward = np.mean(cumul_reward_ / step_)
            precision = np.mean(tp_ / test_rec_round)
            recall = np.mean(tp_ / (self.rela_num[users] + 1e-20))
            metric_list = [ave_reward, precision, recall]
            metric_lists.append(metric_list)
        return metric_lists


    def run(self):
        if self.model_name != None:
            self.model_name = self.epoch2modelname(self.start_epoch)
        else:
            self.model_name = 1
        for i in range(self.start_epoch, self.max_training_step):
            metric_lists1 = self.train()
            self.test_(metric_lists=metric_lists1,stage='train', test_rec_round=self.test_rec_round)
            # self.evaluate(is_test=False)
            metric_lists = self.evaluate(is_test=False)
            self.test_(metric_lists=metric_lists, stage='test',test_rec_round=self.test_rec_round)
            print(self.dqn.gamma)
            print(self.dqn.eps_threshold)
            print(i)

    def test(self):
        self.evaluate()

    def modelname2epoch(self,modelname):
        epoch = math.floor(modelname/math.ceil(self.user_train_num/self.sample_size))
        return epoch

    def epoch2modelname(self,epoch):
        modelname = (epoch * math.ceil(self.user_train_num/self.sample_size)) + 1
        return modelname

    def save_model(self,modelname):
        torch.save({
            'eval_net': self.eval_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'gcn': self.gcn.state_dict(),
            'ssab': self.ssab.state_dict(),
            'DIP': self.DIP.rnns.state_dict(),
            'add': self.add.state_dict(),
            'item_emb': self.item_emb.state_dict(),
            'user_emb': self.user_emb.state_dict(),
            'optimizer': self.dqn.optimizer.state_dict(),
            'attention': self.dqn.att.state_dict(),
            'attention1': self.dqn.att1.state_dict(),
            # 'memory': self.dqn.memory,
            'modelname':modelname,
        }, self.save_path + 's%d' % modelname)

    def modelname2epoch(self,modelname):
        epoch = math.floor(modelname/math.ceil(self.user_train_num/self.sample_size))
        return epoch

    def epoch2modelname(self,epoch):
        modelname = (epoch * math.ceil(self.user_train_num/self.sample_size)) + 1
        return modelname

    def load_model(self):
        checkpoint = torch.load(self.load_path)
        self.eval_net.load_state_dict(checkpoint['eval_net']),
        self.gcn.load_state_dict(checkpoint['gcn'])
        self.ssab.load_state_dict(checkpoint['ssab'])
        self.DIP.rnns.load_state_dict(checkpoint['DIP'])
        # self.add.load_state_dict(checkpoint['add'])
        self.item_emb.load_state_dict(checkpoint['item_emb'])
        self.user_emb.load_state_dict(checkpoint['user_emb'])
        self.att.load_state_dict(checkpoint['attention'])
        self.att1.load_state_dict(checkpoint['attention1'])
        self.start_epoch = self.modelname2epoch(checkpoint['modelname'])

    def load_opt(self):
        checkpoint = torch.load(self.load_path)
        self.target_net.load_state_dict(checkpoint['target_net']),
        # self.dqn.memory = checkpoint['memory']
        self.dqn.optimizer.load_state_dict(checkpoint['optimizer'])
