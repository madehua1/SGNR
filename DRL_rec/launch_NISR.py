# --coding:utf-8--
import argparse
import pickle

import torch.cuda
import os
from DRL_rec.DRLRecommender import Recommender


dataset = 'LastFM'
parser =argparse.ArgumentParser()

#############################################################parameter for NISR########################################################
# """LastFM"""
#env
if dataset == 'LastFM':
    parser.add_argument("--eval_lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--gcn_lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--rnn_lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--ssab_lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--add_lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--item_lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--user_lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--att_lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--l2", type=float, default=1e-3, help="模型的正则化系数")
    #parser.add_argument("--l2", type=float, default=1e-5, help="模型的正则化系数")
    parser.add_argument("--gcn_layer", type=int, default=2, help="图卷积神经网络的层数")




"""路径"""
parser.add_argument('--sort', type=bool, default=False)
parser.add_argument('--dataset_name', type=str, default="%s"%dataset,help="数据集名称")
parser.add_argument("--object_path",type=str,default="../data_LastFM/env_object.pkl",help="环境文件名")
parser.add_argument("--social_path",type=str,default="../data_LastFM/u_u_sorted.dat",help="用户社交关系文件名")
parser.add_argument("--item_path",type=str,default="../data_LastFM/item_dict.pkl",help="读取items")
parser.add_argument("--user_path",type=str,default="../data_LastFM/user_dict.pkl",help="读取users")
parser.add_argument('--model_path', type=str, default="../data_LastFM数据集%s/model/")
parser.add_argument('--result_path', type=str, default="../data_LastFM数据集%s/result/")


parser.add_argument("--test_rec_round",type=int,default=20,help="测试时的推荐轮数")
parser.add_argument("--dataset",type=str,default='implicit',help="数据集")
parser.add_argument("--max_rating",type=int,default=1,help="最大评分")           #LastFM
parser.add_argument("--min_rating",type=int,default=-1,help="最小评分")
parser.add_argument("--boundary_rating",type=float,default=0.5)
parser.add_argument("--episode_length",type=int,default=20)
parser.add_argument("--e_dim",type=int,default=30)
#DQN
parser.add_argument("--tau",type=float,default=0.01)
parser.add_argument("--gamma",type=float,default=0.2)
parser.add_argument("--Eta",type=float,default=0.1)
parser.add_argument("--eps_start",type=float,default=0.5)
parser.add_argument("--eps_end",type=float,default=0.0)
parser.add_argument("--eps_decay",type=float,default=0.005)
parser.add_argument("--memory_size",type=int,default=100000)
parser.add_argument("--test_memory_size",type=int,default=20)
parser.add_argument("--pop",type=int,default=200)
parser.add_argument("--candi_num",type=int,default=100)
parser.add_argument("--fix_emb",type=bool,default=False)
parser.add_argument("--duling",type=bool,default=True)
parser.add_argument("--double_q",type=bool,default=True)
parser.add_argument("--bn_flag",type=bool,default=False)
parser.add_argument("--attention_flag",type=bool,default=True)
parser.add_argument("--preference_flag",type=bool,default=True)
parser.add_argument("--social_flag",type=bool,default=True)
parser.add_argument("--norm",type=bool,default=False)
parser.add_argument("--add_method",type=str,default='None')
#train
parser.add_argument("--method", type=str, default='GRU&GAT', help="方法名称")
parser.add_argument("--al", type=float, default=0.5, help="模型的正则化系数")
parser.add_argument("--lr", type=float, default=0.01, help="学习率")
parser.add_argument("--train_rate",type=float,default=0.8,help="rate for training")
parser.add_argument("--test_rate",type=float,default=0.2,help="rate for testing")
parser.add_argument("--pre_method",type=str,default='rnn',help="选择建模用户兴趣的方法")
parser.add_argument("--max_step",type=int,default=100)
parser.add_argument("--ver_step",type=int,default=1)
parser.add_argument("--update_step",type=int,default=100)
parser.add_argument("--update_times",type=int,default=10,help="update times")
parser.add_argument('--sample_size',type=int,default=10,help="batch size for sample")
parser.add_argument('--test_sample_size',type=int,default=20,help="batch size for sample in testing")
parser.add_argument("--batch_size",type=int,default=128,help="batch size for Dqn learning")
parser.add_argument("--test_batch_size",type=int,default=20,help="batch size for Dqn learning")
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--modelname', type=str, default=None)
parser.add_argument('--save_step', type=int, default=1)
parser.add_argument("--alpha",type=float,default=0.0,help='sequential pattern')
parser.add_argument('--version', type=int, default=1423)



##########################################Train###############################
def train():
    args = parser.parse_args()
    rec =Recommender(args)
    torch.cuda.empty_cache()
    rec.run()

test_rec_round = 20
#test(dataset=dataset,test_rec_round=test_rec_round)
train()



