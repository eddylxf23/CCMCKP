import sys
sys.path.append('.')

import re
import copy
import numpy as np
from src.utils import my_file
from src.utils.factor import Factor
from scipy.stats import norm

class Good():
    def __init__(self, row, col, p, w):
        self.row = row
        self.col = col
        self.p = p
        self.w = w
 
    def __repr__(self):
        return str((self.w, self.p))
 
def goods_filter_LP_dominated(goods, exacting=True):
 
    [goods[i].sort(key=lambda n:n.w, reverse=False) for i in range(len(goods))]
 
    goods1 = [[goods[i][0]] for i in range(len(goods))]
    for i in range(len(goods)):
        p, w = goods1[i][0].p, goods1[i][0].w
        for j in range(1, len(goods[i])):
            if goods[i][j].w == w and goods[i][j].p > p:
                goods1[i][-1] = goods[i][j]
                p = goods[i][j].p
            elif goods[i][j].w > w and goods[i][j].p > p:
                goods1[i].append(goods[i][j])
                p, w = goods[i][j].p, goods[i][j].w
 
    # 严格LP_dominated
    if exacting:
        goods2 = [[goods1[i][0]] for i in range(len(goods1))]
        for i in range(len(goods1)):
            j = 0
            while j < len(goods1[i]) - 1:
                slopes = [[k, (goods1[i][k].p - goods1[i][j].p) / (goods1[i][k].w - goods1[i][j].w)] for k in range(j + 1, len(goods1[i]))]
                slopes.sort(key=lambda x: x[1], reverse=True)
                j = slopes[0][0]
                goods2[i].append(goods1[i][j])
        goods1 = goods2
 
    return goods1
 
 
#每次更新都选择局部提升最大的
def Dynamic(W, P, n2f, c):

    goods = [[Good(i, j, P[i][j], W[i][j]) for j in range(len(W[i]))] for i in range(len(W)) if len(W[i]) > 0]
    G = copy.deepcopy(goods)

    goods = goods_filter_LP_dominated(G, exacting=False)
 
    class Combination():
        def __init__(self, P, W, L):
            self.P = P
            self.W = W
            self.L = L
 
        def __repr__(self):
            return str((self.W, self.P))
 
    C = {goods[0][j].w: Combination(goods[0][j].p, goods[0][j].w, [goods[0][j]]) for j in range(len(goods[0]))}
 
    for i in range(1, len(goods)):
        C_next = {}
        for j in range(len(goods[i])):
            for k,v in C.items():
                if goods[i][j].w + k > c:
                    continue
                #不在或价值更高则更新
                if (goods[i][j].w + k not in C_next) or (goods[i][j].w + k in C_next and goods[i][j].p + v.P > C_next[goods[i][j].w + k].P):
                    C_next[goods[i][j].w + k] = Combination(goods[i][j].p + v.P, goods[i][j].w + k, v.L+[goods[i][j]])
        C = C_next
 
    if len(C) != 0:
        res = list(C.values())
        res.sort(key=lambda x:x.P, reverse=True)
        res_choose = res[0].L
        obj_value = res[0].P
        weight = res[0].W

        solution = []
        for i in range(len(goods)):
            column = res_choose[i].col
            ind_ = n2f[i][column].id
            solution.append(ind_)

        return solution
    
    else:
        return None

def Get_factor(param, data_folder):
    node_num, factor_num= int(param[0]), int(param[1])
    factor_num_sum = node_num * factor_num
    node_id_list = [i for i in range(node_num)]
    factor_id_list = [i for i in range(factor_num_sum)]

    # 读取节点与因子关系
    node_to_factor_id = my_file.load_pkl_in_repo(data_folder, 'node_to_factor.pkl')
    cost_list = np.loadtxt(my_file.real_path_of(data_folder, 'cost.txt'))
    max_cost = np.max(cost_list)

    factor_id2obj = [] # 映射：因子id->具体因子

    for factor_id in factor_id_list:
        sample_filename_ = f'{factor_id}_sample.txt'

        sample_path_ = my_file.real_path_of(data_folder, sample_filename_)

        cost_ = cost_list[factor_id]
        value_ = max_cost - cost_list[factor_id]

        cur_factor = Factor(factor_id, cost_, value_, sample_path_)
        factor_id2obj.append(cur_factor)

    node_to_factors = {node_id : [] for node_id in node_id_list} # list 每个环节对应因子
    for node_id in node_id_list:
        cur_factor_id_list = node_to_factor_id[node_id]

        for cur_factor_id in cur_factor_id_list:
            node_to_factors[node_id].append(factor_id2obj[cur_factor_id])

    value_list = []

    for node_id in node_id_list:

        cur_factors = node_to_factors[node_id]
        cur_factors_value_list = [f.value for f in cur_factors]        
        value_list.append(cur_factors_value_list)
    
    return value_list, node_to_factors, factor_id2obj

def Get_normal_mu_variance(node_to_factors):
    node_num = len(node_to_factors)
    node_id_list = [i for i in range(node_num)]
    n2f_mean = {node_id : [] for node_id in node_id_list}
    n2f_variance = {node_id : [] for node_id in node_id_list}

    for node_id in node_id_list:

        cur_factors = node_to_factors[node_id]
        for f in cur_factors:
            mu, std = norm.fit(f.samples)
            n2f_mean[node_id].append(mu)
            n2f_variance[node_id].append(std**2)

    return n2f_mean, n2f_variance

def Get_weight(node_to_factors, node_to_factors_mean, node_to_factors_variance, _lambda):

    node_num = len(node_to_factors)
    node_id_list = [i for i in range(node_num)]

    weight_list = []

    for node_id in node_id_list:

        cur_factors = node_to_factors[node_id]
        cur_factors_mean_list = node_to_factors_mean[node_id]
        cur_factors_var_list = node_to_factors_variance[node_id]
        lambda_va = [x * _lambda for x in cur_factors_var_list]
        
        # 样本均值和样本标准差按权重相加
        weight_factors_list = [cur_factors_mean_list[i] + lambda_va[i] for i in range(len(cur_factors))]

        weight_list.append(weight_factors_list)
    
    return weight_list

if __name__ == '__main__':
    # 物品价值P
    # 物品重量W
    # 背包容量Tmax

    folder_name = 'benchmark/Instance_lab_4_4_[2, 5, 10, 20, 50, 100, 500, 1000]_'
    param = re.findall(r'\d+',folder_name)
    Tmax = 14
    CL = 0.9
    MaxIter_LS = 30
    MonteCarloTimes = 100000
    #samplesize_list = [2,5,10,20,50,100,500,1000]
    samplesize = 100
    _lambda = 1
    
    P,n2f, f2o = Get_factor(param, samplesize, folder_name)
    n2f_m, n2f_v = Get_normal_mu_variance(n2f)
    W = Get_weight(n2f, n2f_m, n2f_v, _lambda)
    solution, obj_Dynamic, weight_Dynamic = Dynamic(W, P, n2f, Tmax)

    print("solution id:", solution)
    print("obj_Dynamic:", obj_Dynamic)
    print("weight_Dynamic:", weight_Dynamic)