'''
Constru_Proce.py
生成局部搜索的初始解

factor_id2obj: 自定义Factor类
'''
import re
import sys
sys.path.append(sys.path[0] + '/../')
import random
import heapq
import numpy as np
from multiprocessing import Pool
from src.functions.func_solution_factor import solution_cost, Solution_Evaluation, Solution_Initialization, Statistic_factor, confidence_list_evaluation
from src.functions.func_cl_estimator import monte_carlo_estimate, advanced_monte_carlo, exact_evaluation, advanced_exact_evaluation,factorization_list
from src.functions.func_realP_statistical import real_P_test

def Construct_Procedure(Eval_func, node_resort_w, node_resort_u, fi, threshold_p, T_max, param, MC_sample_size, quick_check_list):

    Eval_times = 0
    Solution = Solution_Initialization(node_resort_u, param, 0)
    _St = np.copy(Solution)
    available_S = np.copy(Solution)
    _pt = Solution_Evaluation(Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)
    stopping_condition = -np.ones(len(available_S),int)
    select_node = -1
    Eval_times += 1
    Tem_solution = [[solution_cost(available_S,fi),_pt,available_S,Eval_times]]
    while 1:

        factor_list = []
        for factor_id in _St:
            factor_list.append(fi[factor_id])

        solution_weight = [factor.weight for factor in factor_list]

        for i in range(len(available_S)):
            if available_S[i] == -1:
                solution_weight[i] = -1

        select_node = np.argsort(solution_weight)[-1] # 当前解中weight最大的factor所在的node
        # print(select_node)
        select_factor = _St[select_node]
        # print(select_factor)

        # 找到weight矩阵中比factor小的下一个因子，对应到utility矩阵中的序号
        if node_resort_w[select_node].index(select_factor) < len(node_resort_w[select_node])-1: 

            _St[select_node] = node_resort_w[select_node][node_resort_w[select_node].index(select_factor)+1]
            _pt = Solution_Evaluation(Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)
            c_ = solution_cost(_St,fi)
            print("当前解：",c_ ,_pt,_St)
            Eval_times += 1
            Tem_solution.append([c_ ,_pt,_St,Eval_times])
            if _pt >= threshold_p:
                # print("Yeah!")
                break
        else:
            # 随后的循环不再替换本次的select_node
            available_S[select_node] = -1
            if (available_S == stopping_condition).all():
                print("Warning: 没有找到可行解, 初始解设为每个class下最小weight的item!")
                _St = Solution_Initialization(node_resort_w, param, -1)
                break   

    init_feasible_Solution = _St
    init_obj_cost = solution_cost(_St,fi)
    init_p = _pt

    return init_feasible_Solution, init_obj_cost, init_p, Eval_times, Tem_solution


if __name__=='__main__':

    sign = 'lab'
    Instance_folder = [\
        'benchmark/Instance_lab_5_10_30_',
        'benchmark/Instance_lab_10_20_500_']  
    
    IF = Instance_folder[0]
    param = re.findall(r'\d+',IF)
    threshold_p = 0.99
    Evaluation =  advanced_monte_carlo
    MaxIter_LS = 30
    MC_sample_size = 100000
    
    _lambda = 2
    see = np.random.seed(233)

    

    
