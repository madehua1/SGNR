"""完成环境的编写"""
import pickle
import random

import torch
import torch as th
import torch.utils.data as data
import numpy as np
import copy

class Env():
    def __init__(self,episode_length=32,alpha=0.0,boundary_rating=0.5,max_rating=1,min_rating=-1,env_path=None, method='NISR',dataset='implicit',delta=None,item_emb=None):
        self.episode_length = episode_length
        self.alpha = alpha
        self.boundary_rating = boundary_rating
        self.max_rating = max_rating
        self.min_rating = min_rating
        self.a = 2.0 / (float(max_rating) - float(min_rating))
        self.b = - (float(max_rating) + float(min_rating)) / (float(max_rating) - float(min_rating))
        self.positive =self.a * self.boundary_rating +self.b
        self.env_path = env_path
        object = pickle.load(open(self.env_path,"rb"))
        self.rating_mat = object["rating_mat"]
        self.user_num = object["user_num"]
        self.item_num = object["item_num"]
        self.rela_num = object["rela_num"]
        self.method = method
        self.dataset = dataset
        self.delta = delta
        self.item_emb = item_emb

    def get_init_data(self):
        return self.user_num, self.item_num, self.rela_num

    def calculate_r(self,item_id):
        # delta = self.delta[self.user_id]
        # item_id = th.tensor(item_id,device='cuda',dtype=th.long)
        # if len(self.state) > 0:
        #     state = th.tensor(self.state,device='cuda',dtype=th.long)
        #     qj = self.item_emb(state)
        #     qi = self.item_emb(item_id).unsqueeze(dim=0)
        #     C = th.sum(qi * qj, dim=1)
        #     D = th.sum(1 - C)
        #     p = delta * self.rating_mat[self.user_id, item_id] + (1-delta) * D /len(self.state)
        # else:
        #     p = self.rating_mat[self.user_id, item_id]
        p = self.rating_mat[self.user_id, item_id]
        # if p > self.boundary_rating:
        #     p = 1
        # else:
        #     p = 0
        # if torch.is_tensor(p):
        #     p = p.item()
        # print(p)
        # r = self.rating_mat[self.user_id, item_id]
        # if r > 0:
        #     r = 1
        # if r < 0:
        #     r = 0
        return p

    def reset(self,user_id):
        self.user_id = user_id
        self.step_count = 0
        self.con_neg_count = 0
        self.con_pos_count = 0
        self.con_zero_count = 0
        self.con_not_neg_count = 0
        self.con_not_pos_count = 0
        self.all_neg_count = 0
        self.all_pos_count = 0
        self.history_items = set()
        self.state = []
        if self.method == 'NICF':
            self.state1 = {}
            if self.dataset == 'explicit':
                for i in range(6):
                    self.state1[i] = []
                return self.state,self.state1
            if self.dataset == 'implicit':
                for i in range(2):
                    self.state1[i] = []
                return self.state,copy.deepcopy(self.state1)
        return copy.deepcopy(self.state)

    def step(self, item_id,test_rec_round=None):
        '''
        模拟用户与item的交互
        :param item_id: item的id
        :return: 当前的状态，当前item的奖励和交互状态（是否交互结束）
        '''
        reward = [0.0, False]
        # r = self.rating_mat[self.user_id, item_id]
        # # normalize the reward value
        # if r == 0:
        #     reward[0] = 0
        # else:
        #     if r >  0:
        #         reward[0] = r
        #     else:
        #         reward[0] = 0
        #     #reward[0] = self.a * r + self.b
        r = self.calculate_r(item_id=item_id)
        reward[0] = self.a * r + self.b
        #reward[0] = r
        # if r != 0:
        #     reward[0] = self.a * r + self.b
        # if r == 0:
        #     reward[0] = 0
        #reward[0] = r
        self.step_count += 1
        sr = self.con_pos_count - self.con_neg_count

        reward[0] += self.alpha * sr

        if reward[0] < self.positive:
        #if reward[0] < 0:
            self.con_neg_count += 1
            self.all_neg_count += 1
            self.con_not_pos_count += 1
            self.con_pos_count = 0
            self.con_not_neg_count = 0
            self.con_zero_count = 0
        elif reward[0] > self.positive:
        #elif reward[0] > 0:
            self.con_pos_count += 1
            self.all_pos_count += 1
            self.con_not_neg_count += 1
            self.con_neg_count = 0
            self.con_not_pos_count = 0
            self.con_zero_count = 0
        else:
            self.con_not_neg_count += 1
            self.con_not_pos_count += 1
            self.con_zero_count += 1
            self.con_pos_count = 0
            self.con_neg_count = 0

        self.history_items.add(item_id)
        if test_rec_round != None:
            if self.step_count == test_rec_round or len(self.history_items) == self.item_num:
                reward[1] = True
        else:
            if self.step_count == self.episode_length or len(self.history_items) == self.item_num:
            # 到达episode最大长度或者选择了所有的item
                reward[1] = True

        if r > self.boundary_rating:
            self.state.append(item_id)
        # if r > self.boundary_rating:
        #     self.state.append(item_id)
        curs = copy.deepcopy(self.state)
        if self.method == 'NICF':
            if self.dataset == 'explicit':
                self.state1[int(r)].append(item_id)
                curs1 = copy.deepcopy(self.state1)
                return curs,curs1,r,reward[0],reward[1]
            if self.dataset == 'implicit':
                # if r < self.boundary_rating: r1=0
                # else: r1=1
                # self.state1[int(r1)].append(item_id)

                if r <= self.boundary_rating:
                #if reward[0] <= self.positive:
                    r1=0
                    self.state1[int(r1)].append(item_id)
                if r > self.boundary_rating:
                #if reward[0] > self.positive:
                    r1=1
                    self.state1[int(r1)].append(item_id)
                # if r == 0:
                #     r1 = None
                curs1 = copy.deepcopy(self.state1)
                return curs, curs1, r1, reward[0], reward[1]
        return curs, reward[0], reward[1]



