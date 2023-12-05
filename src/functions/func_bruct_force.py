'''
Bruct_force.py

bruct_force: 暴力搜索
'''
import sys
sys.path.append(sys.path[0] + '/../')
sys.path.append('..')

import time
import numpy as np
import itertools
import operator
import re
from functions.func_solution_factor import solution_cost,Statistic_factor,Solution_Evaluation
from functions.func_cl_estimator import monte_carlo_estimate, advanced_monte_carlo, exact_evaluation, advanced_exact_evaluation,Hoeffding_estimate,Bernstein_estimate, factorization_list

def bruct_force(Eval_func,node_resort_v,fi, CL, T_max,param,MC_sample_size, quick_check_list):
 
    node_id_list = [i for i in range(int(param[0]))] 
    all_sol_list = []
    _St,S_opt = [],[]
    _ct,Eval_times,_pt,_p = 0,0,-1,-1
    c_opt = float("inf")
    feasible_count = 0

    # if CL >=0.99999:
    #     threshold_p = CL-0.000005
    # elif CL>=0.9999:
    #     threshold_p = CL-0.00005
    # elif CL >=0.999:
    #     threshold_p = CL-0.0005
    # elif CL >=0.99:
    #     threshold_p = CL-0.005
    # else:
    #     threshold_p = CL-0.05

    # 列出所有的可能解
    select_pool = range(int(param[1]))
    all_sol_list = []

    for selected_cand_idx in itertools.product(select_pool, repeat=int(param[0])):
        cur_solution = np.zeros(len(node_id_list), dtype=int)

        for node_id in node_id_list:
            _factor_id = node_resort_v[node_id][selected_cand_idx[node_id]]
            cur_solution[node_id] = _factor_id

        all_sol_list.append(cur_solution)

    # 对每一个解进行评估，取目标函数值最大的解
    maxp = 0
    minp = 1
    for solution in all_sol_list:

        start_t = time.time()
        print(solution)
        _pt = Solution_Evaluation(Eval_func, solution, T_max, fi, CL, param, MC_sample_size, quick_check_list)
        _ct = solution_cost(solution,fi)
        if _pt:
            Eval_times += 1

        maxp = max(maxp,_pt)
        minp = min(minp,_pt)

        print(_pt,_ct,"Time:",time.time()-start_t)
        
        if _pt >= CL:
            # print(solution,_pt)
            feasible_count += 1
            _St = np.copy(solution)
            _ct = solution_cost(solution,fi)
            if _ct < c_opt:
                S_opt = np.copy(_St)
                c_opt = _ct
                _p = _pt

    Optimal_Solution = S_opt
    Optimal_cost = c_opt
    Optimal_p = _p
    if _p == -1:
        print("There is no feasible solution under this condition!")
    
    return  Optimal_Solution, Optimal_cost, Optimal_p ,Eval_times, feasible_count, maxp, minp

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
    folder_name = 'benchmark/Mix_Instance_lab_4_5_[500]_'
    param = re.findall(r'\d+',folder_name)
    Tmax = 10
    CL = 0.9
    _lambda = 2
    samplesize = int(param[2])
    MC_sample_size = 100000
    Evaluation = monte_carlo_estimate
    Aw,Au,Av,FactortoObject = Statistic_factor(param,  _lambda, folder_name)
    # quick_check_list = factorization_list(samplesize,int(param[0]),CL,10,20) 
    quick_check_list=[]
    print("---- 开始暴力搜索！")
    Solution = bruct_force(Evaluation,Av, FactortoObject, CL, Tmax, param, MC_sample_size, quick_check_list)
    print("Solution:",Solution[0],"cost:",Solution[1],"confidence level:",Solution[2],"Evaluation times:",Solution[3], "Feasible:",Solution[4], "maxp:",Solution[5], "minp:",Solution[6] )