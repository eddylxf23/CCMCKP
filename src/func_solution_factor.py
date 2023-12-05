'''
Solution_factor_func.py

生成解和factor相关操作
'''
import sys
sys.path.append(sys.path[0] + '/../')
sys.path.append('..')
import time
import numpy as np
from utils import my_file
from utils.factor import Factor
from functions.func_cl_estimator import monte_carlo_estimate, advanced_monte_carlo, exact_evaluation, advanced_exact_evaluation,Hoeffding_estimate,Bernstein_estimate

def Statistic_factor(param, _lambda,data_folder):
    '''
    Statistic_factor: 给出所有item的必要统计信息及重排序

    wf:权重系数lambda
    param: benchmark系数,[class数,item数,样本数,0,整数的CL]
    data_folder: benchmark数据存放文件夹

    输出: 按weight排序的Item, 按utility排序的Item, 按value排序的Item, Factor类的所有Item
    '''
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
        big_sample_filename_ = f'{factor_id}_big_sample.txt'

        sample_path_ = my_file.real_path_of(data_folder, sample_filename_)
        big_sample_path_ = my_file.real_path_of(data_folder, big_sample_filename_)

        cost_ = cost_list[factor_id]
        value_ = max_cost - cost_list[factor_id]
        # big_sample = np.loadtxt(big_sample_path_)
        # big_samples_list.append(big_sample)

        cur_factor = Factor(factor_id, cost_, value_, sample_path_, big_sample_path_)
        factor_id2obj.append(cur_factor)

    print(f'\n====================== Read Complete! {len(factor_id2obj)} Factors ===================')

    node_to_factors = {node_id : [] for node_id in node_id_list} # list 每个环节对应因子
    for node_id in node_id_list:
        cur_factor_id_list = node_to_factor_id[node_id]

        for cur_factor_id in cur_factor_id_list:
            node_to_factors[node_id].append(factor_id2obj[cur_factor_id])

    node_resort_utility = {node_id: [] for node_id in node_id_list}
    node_resort_weight = {node_id: [] for node_id in node_id_list}
    node_resort_value = {node_id: [] for node_id in node_id_list}

    for node_id in node_id_list:

        cur_factors = node_to_factors[node_id]
        cur_factors_mean_list = [f.sample_mean for f in cur_factors]
        # print("mean:",cur_factors_mean_list)
        cur_factors_std_list = [f.sample_std for f in cur_factors]
        # print("std:",cur_factors_std_list)
        cur_factors_value_list = [f.value for f in cur_factors]
        cur_factors_std_list[:] = [x * _lambda for x in cur_factors_std_list]
        
        # 样本均值和样本标准差按权重相加
        weight_factors_list = [cur_factors_mean_list[i] + cur_factors_std_list[i] for i in range(len(cur_factors))]
        utility_factors_list = [cur_factors_value_list[i] / weight_factors_list[i] for i in range(len(cur_factors))]

        for i in range(len(weight_factors_list)):
            cur_factors[i].set_weight(weight_factors_list[i])
            cur_factors[i].set_utility(utility_factors_list[i])

        cand_indices_weight = np.argsort(weight_factors_list)[::-1]    # 根据统计量构造的 weight=mean+lambda*std 从大到小 将每个class的因子重新排序
        cand_indices_utility = np.argsort(utility_factors_list)[::-1]    # 根据统计量构造的 utility=value/weight 从大到小 将每个class的因子重新排序
        cand_indices_value = np.argsort(cur_factors_value_list)[::-1]   # 根据 value 构造的 从大到小 将每个class的因子重新排序

        for _idx in cand_indices_weight:
            node_resort_weight[node_id].append(cur_factors[_idx].id)

        for _idx in cand_indices_utility:
            node_resort_utility[node_id].append(cur_factors[_idx].id)

        for _idx in cand_indices_value:
            node_resort_value[node_id].append(cur_factors[_idx].id)

    return node_resort_weight, node_resort_utility, node_resort_value, factor_id2obj

def Solution_Initialization(node_resort,param,order):

    node_id_list = [i for i in range(int(param[0]))]   
    Init_solution = np.zeros(len(node_id_list), dtype=int)

    for node_id in node_id_list:
        # 取出按照排好序的每个node中的第一个factor，即对应属性最大的factor
        Init_solution[node_id] = node_resort[node_id][order]
    
    return Init_solution


def Solution_Evaluation(eval_func, Solution, T_max, factor_id2obj, CL, param, MC_sample_size, quick_check_list):

    factor_list = []
    for factor_id in Solution:
        factor_list.append(factor_id2obj[factor_id])

    return eval_func(factor_list, T_max, MC_sample_size, CL, param, quick_check_list)

def confidence_list_evaluation(ConfidencRecord, Eval_func, solution, T_max, fi, CL, param, MC_sample_size, quick_check_list):
    # c_array格式：np.array([[eval_times,average_p,solution[0],...,solution[12]],...,[eval_times,average_p,solution[0],...,solution[12]]])
    # 评估次数：ConfidencRecord[:][0]
    # 平均置信度： ConfidencRecord[:][1]
    # 解： ConfidencRecord[:][2:]
    _index = np.where(np.equal(ConfidencRecord[:,2:],solution).all(1)==True)[0]

    if _index.size:
        # 在c_array中，找出Index并检查eval_times
        s_index = _index[0]
        if ConfidencRecord[s_index][0] == 3:
            Eval_times = 1
            return ConfidencRecord[s_index][1],Eval_times
        
        else:
            p = Solution_Evaluation(Eval_func, solution, T_max, fi, CL, param, MC_sample_size, quick_check_list)
            ConfidencRecord[s_index][1] = (ConfidencRecord[s_index][0]*ConfidencRecord[s_index][1] + p)/(ConfidencRecord[s_index][0]+1)
            ConfidencRecord[s_index][0] += 1
            Eval_times = 1
            return ConfidencRecord[s_index][1],Eval_times
        

    else:
        # 不在c_array中，增添
        p = Solution_Evaluation(Eval_func, solution, T_max, fi, CL, param, MC_sample_size, quick_check_list)
        s_list = [1,p]
        s_list.extend(solution)
        ConfidencRecord = np.vstack((ConfidencRecord,s_list))
        Eval_times = 1
        
        return p, Eval_times
    

def solution_cost(solution,factor_id2obj):
    cost = 0
    for _factor_id in solution:
        _factor = factor_id2obj[_factor_id]
        cost += _factor.cost
    
    return cost