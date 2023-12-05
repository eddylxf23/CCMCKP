'''
Greedy.py

greedy: 贪心算法, 以最大value为始终目标
'''
import sys
sys.path.append(sys.path[0] + '/../')
sys.path.append('..')

import numpy as np
import re
import random
from functions.func_solution_factor import solution_cost,Statistic_factor,Solution_Evaluation
from functions.func_cl_estimator import monte_carlo_estimate, advanced_monte_carlo, exact_evaluation, advanced_exact_evaluation,Hoeffding_estimate,Bernstein_estimate, factorization_list
from functions.func_realP_statistical import real_P_test 

def Greedy(Eval_func, node_resort_w, fi, threshold_p, T_max, param, MC_sample_size, quick_check_list):
    Eval_times = 0
    node_id_list = [i for i in range(int(param[0]))]   
    Init_solution = np.zeros(len(node_id_list), dtype=int)

    node_resort_i = {node_id: [] for node_id in node_id_list}
    for node_id in node_id_list:

        cur_factors = node_resort_w[node_id][::-1] # 令weight从小到大排序
        cur_factors_weight_list = [fi[factors].weight for factors in cur_factors]
        cur_factors_value_list = [fi[factors].value for factors in cur_factors]
        
        increment_factors_list = [(cur_factors_value_list[i] -cur_factors_value_list[i-1])/ (cur_factors_weight_list[i]-cur_factors_weight_list[i-1]) for i in range(1,len(cur_factors))]
        cand_indices_increment = np.argsort(increment_factors_list)[::-1]  # 根据统计量构造的 increment 从大到小 将每个class的因子重新排序（除了weight最小的那一个）
        node_resort_i[node_id].append(node_resort_w[node_id][-1])

        for _idx in cand_indices_increment:
            node_resort_i[node_id].append(fi[cur_factors[_idx + 1]].id) # 此处构造的列表缺少最小的weight

        for i in range(len(increment_factors_list)):
            fi[cur_factors[i+1]].set_increment(increment_factors_list[i]) # 跳过第一个

        Init_solution[node_id] = node_resort_i[node_id][0]

    # 初始化解以及各个node的候选factor
    _St = np.copy(Init_solution)
    _S =  np.copy(Init_solution)
    _pt = Solution_Evaluation(Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)
    pp = _pt
    c_opt = solution_cost(_St,fi)
    Eval_times += 1
    Tem_opt_solution = [[c_opt,_pt,_St,Eval_times]]

    candidate_factor_number = [1 for _ in range(len(_St)) if len(node_resort_i[node_id]) > 1]
    candidate_factor_increment = [node_resort_i[node_id][1] for node_id in node_id_list if len(node_resort_i[node_id]) > 1]
    
    print(candidate_factor_increment)
    # print("St:",_St,_pt)

    increment_factors_list_temp = []
    for factor_id in candidate_factor_increment:
        increment_factors_list_temp.append(fi[factor_id].increment)    # 候选factor构造factor increment list
    
    _pt = -1

    while 1:
        # print(candidate_factor_increment)
        factor_id_max = candidate_factor_increment[np.argmax(increment_factors_list_temp)]     # 找出候选factor list中最大increment的factor
        node_id_max = np.argmax(increment_factors_list_temp)    # 最大increment的factor的node
        # print(factor_id_max,node_id_max)

        _S[node_id_max] = factor_id_max      # 将最大increment的factor替换到解S对应node的factor
        _p = Solution_Evaluation(Eval_func, _S, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)
        Eval_times += 1
        # print("_S:",_S,_p)

        if _p < threshold_p or (_St == _S).all():
            break           
        else:
            _St = np.copy(_S)
            _pt = _p
            _ct = solution_cost(_St,fi)
            # 更新 candidate list
            if candidate_factor_number[node_id_max]<int(param[1])-1:
                candidate_factor_number[node_id_max] += 1
            candidate_factor_increment[node_id_max] = node_resort_i[node_id_max][candidate_factor_number[node_id_max]]
            increment_factors_list_temp[node_id_max]= fi[candidate_factor_increment[node_id_max]].increment
            Tem_opt_solution.append([_ct,_pt,_St,Eval_times])
             
    Greedy_Solution = np.copy(_St)
    Greedy_cost = solution_cost(_St,fi)
    Greedy_p = _pt
    
    if _pt == -1:
        Greedy_p = pp
        print("There is no feasible solution under this condition!")
    
    return  Greedy_Solution, Greedy_cost, Greedy_p ,Eval_times, Tem_opt_solution

if __name__=='__main__':
    '''
    评估方法一共有: 
    monte_carlo_estimate, 
    advanced_monte_carlo, 
    exact_evaluation, 
    advanced_exact_evaluation, 
    Hoeffding_estimate, 
    Bernstein_estimate
    '''
    folder_name = 'benchmark/Instance_lab_3_5_30_'
    param = re.findall(r'\d+',folder_name)
    T_m,CL = 14,0.99
    se = random.seed(827)
    MC_sample_size = 1000000
    Evaluation = advanced_exact_evaluation
    quick_check_list = factorization_list(int(param[2]),int(param[0]),CL,10,20)
    Aw,Au,Av,fi = Statistic_factor(param, 2, folder_name)
    print("开始贪心搜索！")
    Solution = Greedy(Evaluation, Aw, fi, CL, T_m, param, MC_sample_size, quick_check_list)
    print("Greedy:",Solution[0],Solution[1],Solution[2],real_P_test(Solution[0], T_m, se, folder_name, 'huawei'),Solution[3])
