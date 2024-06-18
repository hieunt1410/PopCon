import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
from collections import Counter
import shutil

def get_ub(dataname, task):
    with open(os.path.join('../data/{}'.format(dataname
                                               ), 'user_bundle_{}.txt'.format(task)), 'r') as f:
        u_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

    indice = np.array(u_b_pairs, dtype=np.int32)
    values = np.ones(len(u_b_pairs), dtype=np.float32)
    u_b_graph = sp.coo_matrix(
        (values, (indice[:, 0], indice[:, 1])), shape=(8039, 4771)).tocsr()

    return u_b_pairs, u_b_graph

def resplit(dataname):
    train = pd.read_csv(f'../data/{dataname}/user_bundle_train.txt', sep='\t', names=['user', 'bundle'])
    tune = pd.read_csv(f'../data/{dataname}/user_bundle_tune.txt', sep='\t', names=['user', 'bundle'])
    test = pd.read_csv(f'../data/{dataname}/user_bundle_test.txt', sep='\t', names=['user', 'bundle'])
    total = pd.concat([train, tune, test])
    
    path = '../data_pkl/{}'.format(dataname)
    if not os.path.exists(path):
        os.makedirs(path)
    
    with open(f'../data_pkl/{dataname}/user_bundle_train.txt', 'w') as f1, \
    open(f'../data_pkl/{dataname}/user_bundle_tune.txt', 'w') as f2, \
    open(f'../data_pkl/{dataname}/user_bundle_test.txt', 'w') as f3:
        for u in total['user'].unique():
            bundles = total[total['user'] == u]['bundle'].values
            for bundle in bundles[:-2]:
                f1.write(f'{u}\t{bundle}\n')
            f2.write(f'{u}\t{bundles[-2]}\n')
            f3.write(f'{u}\t{bundles[-1]}\n')       
    
    shutil.copy(f'../data/{dataname}/user_item.txt', f'../data_pkl/{dataname}/user_item.txt')
    shutil.copy(f'../data/{dataname}/bundle_item.txt', f'../data_pkl/{dataname}/bundle_item.txt')
    
    
def regen(dataname):
    ub_pairs_train, ub_graph_train = get_ub(dataname, 'train')
    ub_pairs_valid, ub_graph_valid = get_ub(dataname, 'tune')
    ub_pairs_test, ub_graph_test = get_ub(dataname, 'test')
    ub = ub_graph_train + ub_graph_valid + ub_graph_test
    
    # row, col = ub.nonzero()
    
    # train_rows, train_cols = ub.nonzero()[0][:-2], ub.nonzero()[1][:-2]
    # valid_rows, valid_cols = ub.nonzero()[0][-2:-1], ub.nonzero()[1][-2:-1]
    # test_rows, test_cols = ub.nonzero()[0][-1:], ub.nonzero()[1][-1:]
    
    # ub_train = sp.csr_matrix((np.ones(len(train_rows)), (train_rows, train_cols)), shape=ub.shape)
    # ub_valid = sp.csr_matrix((np.ones(len(valid_rows)), (valid_rows, valid_cols)), shape=ub.shape)
    # ub_test = sp.csr_matrix((np.ones(len(test_rows)), (test_rows, test_cols)), shape=ub.shape)
    
    # # return ub_train, ub_valid, ub_test
    # path = '../data_pkl/{}'.format(dataname)
    # if not os.path.exists(path):
    #     os.makedirs(path)
        
    # with open(os.path.join(path, 'user_bundle_train.pkl'), 'wb') as f:
    #     pickle.dump(ub_train, f)
    
    # with open(os.path.join(path, 'user_bundle_tune.pkl'), 'wb') as f:
    #     pickle.dump(ub_valid, f)
        
    # with open(os.path.join(path, 'user_bundle_test.pkl'), 'wb') as f:
    #     pickle.dump(ub_test, f)
        
    # neg = []
    # for u in range(8039):
    #     neg.append(np.random.choice(np.setdiff1d(np.arange(4771), ub_graph_train[u].nonzero()[1]), 99, replace=False))
    # neg = np.array(neg)
    
    # with open(os.path.join(path, 'neg.pkl'), 'wb') as f:
    #     pickle.dump(neg, f)
        
    # print('Done')