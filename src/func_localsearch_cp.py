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

def Construct_Procedure_random(Eval_func, T_max, fi, node_resort_w, threshold_p, param, MC_sample_size, quick_check_list):
    Eval_times = 0
    node_num = int(param[0])
    _St = []
    Tem_solution = []
    for node_index in range(node_num):
        _St.append(random.choice(node_resort_w[node_index]))

    _cost = solution_cost(_St,fi)
    _p = Solution_Evaluation(Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)
    Eval_times += 1

    node_list = [i for i in range(int(param[0]))]
    # 如果随机得到的是不可行解，则不断循环缩小随机的范围，最坏情况下只有最小weight可行
    while _p < threshold_p:
        for node_id in node_list:
            select_factor = _St[node_id]
            # 从weight矩阵中比factor小的所有因子随机选一个
            index_ = node_resort_w[node_id].index(select_factor)
            _St[node_id] = random.choice(node_resort_w[node_id][index_:])
        
        _cost = solution_cost(_St,fi)
        _p = Solution_Evaluation(Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)
        Eval_times += 1

    Tem_solution = [[_cost,_p,_St,Eval_times]]

    return _St,_cost,_p, Eval_times,Tem_solution

def Construct_Procedure_repair(Eval_func, T_max, fi, node_resort_w, threshold_p, param, MC_sample_size, quick_check_list):
    Eval_times = 0
    node_num = int(param[0])
    Single = []
    Tem_solution = []
    for node_index in range(node_num):
        Single.append(random.choice(node_resort_w[node_index]))
    _cost,_p,SingleRepair,Eval_times = RepairSolution(Eval_func, Single, T_max, fi, node_resort_w, threshold_p, param, MC_sample_size, quick_check_list)
    Tem_solution = [[_cost,_p,SingleRepair,Eval_times]]

    return SingleRepair,_cost,_p, Eval_times,Tem_solution

# 修复模块
def RepairSolution(Eval_func, Solution, T_max, fi, node_resort_w, threshold_p, param, MC_sample_size, quick_check_list):
    _St = np.copy(Solution)
    available_S = np.copy(Solution)
    Eval = 0
    stopping_condition = -np.ones(len(available_S),int)
    select_node = -1
    _pt = Solution_Evaluation(Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)
    Eval += 1

    if _pt < threshold_p:
        while 1:
            factor_list = []
            for factor_id in _St:
                factor_list.append(fi[factor_id])

            # solution_utility = [factor.utility for factor in factor_list]
            solution_weight = [factor.weight for factor in factor_list]
            for i in range(len(available_S)):
                if available_S[i] == -1:
                    # solution_utility[i] = np.inf
                    solution_weight[i] = -1

            # select_node = np.argsort(solution_utility)[0] # 当前解中utility最小的factor所在的node
            select_node = np.argsort(solution_weight)[-1] # 当前解中weight最大的factor所在的node
            select_factor = _St[select_node]
    
            if node_resort_w[select_node].index(select_factor) < len(node_resort_w[select_node])-1:     # 查询weight矩阵中比当前factor小的下一个因子
                _St[select_node] = node_resort_w[select_node][node_resort_w[select_node].index(select_factor)+1]          
                _pt = Solution_Evaluation(Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)  # 评估是否满足约束
                Eval += 1
                if _pt >= threshold_p:
                    break
            else:  
                available_S[select_node] = -1   # 如果不再有更小的weight，随后的循环不再替换本次的select_node
                if (available_S == stopping_condition).all():
                    print("Warning: 没有找到可行解, 设为每个class下最小weight的item!")
                    _St = Solution_Initialization(node_resort_w, param, -1)
                    break   
        _St = Solution_Initialization(node_resort_w, param, -1)
        _pt = Solution_Evaluation(Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)  # 评估是否满足约束
        Eval += 1
    return solution_cost(_St,fi), _pt, list(np.copy(_St)), Eval

# 
def Construct_Procedure_A(Eval_func, node_resort_w, node_resort_u, fi, threshold_p, T_max, param, MC_sample_size, quick_check_list):
    Eval_times = 0
    Solution = Solution_Initialization(node_resort_u, param, 0)
    _St = np.copy(Solution)
    available_S = np.copy(Solution)
    _pt = Solution_Evaluation(Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)
    Eval_times += 1
    stopping_condition = -np.ones(len(available_S),int)
    select_node = -1
    print("初始：",Solution)
    sig = 0
    while 1:
        factor_list = []
        for factor_id in _St:
            factor_list.append(fi[factor_id])

        solution_weight = [factor.weight for factor in factor_list]
        for i in range(len(available_S)):
            if available_S[i] == -1:
                solution_weight[i] = -1

        select_node = np.argsort(solution_weight)[::-1][0] # 当前解中weight最大的factor所在的node
        select_factor = _St[select_node]
        # 找到weight矩阵中比factor小的所有因子，对应到utility矩阵中的序号
        for factor_id in node_resort_w[select_node][node_resort_w[select_node].index(select_factor)+1:]: 

            # 找出当前因子在weight排序矩阵中的位置，然后从比当前因子weight小的因子开始依次尝试
            _St[select_node] = factor_id
            _pt = Solution_Evaluation(Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)
            Eval_times += 1
            print("当前解：",solution_cost(_St,fi),_pt,_St)
            if _pt >= threshold_p:
                print("Yeah!")
                sig = 1
                break

        if sig == 1:
            break

        # 随后的循环不再替换本次的select_node
        available_S[select_node] = -1
        if (available_S == stopping_condition).all():
            print("Warning: 没有找到可行解, 初始解设为每个class下最小weight的item!")
            _St = Solution_Initialization(node_resort_w, param, -1)
            break

    init_feasible_Solution = _St
    init_obj_cost = solution_cost(_St,fi)
    init_p = _pt
    # print(f"The Evaluation times of CP_A is {Eval_times}")

    return init_feasible_Solution, init_obj_cost, init_p, Eval_times

def Construct_Procedure_B(Eval_func, node_resort_w, node_resort_u, fi, threshold_p, T_max, param, MC_sample_size, quick_check_list):

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
    if sign == 'lab':
        Instance_folder = [\
            'benchmark/Instance_lab_5_10_30_',
            'benchmark/Instance_lab_10_20_500_']  
    else:
        Instance_folder = [\
            'benchmark/Instance_huawei_5_10_30_',
            'benchmark/Instance_huawei_10_20_500_']
    
    IF = Instance_folder[0]
    param = re.findall(r'\d+',IF)
    threshold_p = 0.99
    Evaluation =  advanced_monte_carlo
    MaxIter_LS = 30
    MC_sample_size = 100000
    
    _lambda = 2
    see = np.random.seed(233)

    Aw,Au,Av,fi = Statistic_factor(param,  _lambda , IF)
    # quick_check_list = factorization_list(int(param[2]),int(param[0]),threshold_p,10,20)

    # M = 40
    # Tmin = [i/2 for i in range(M-20,M)]   
    # Tmiddle = [i/2 for i in range(M,M+20)]
    # Tmax = [i/2 for i in range(M+20,M+60)]
    # # T_m = 47.5
    # # S_cp1 = Construct_Procedure_A(Evaluation, Aw,Au,fi, CL, T_m, param, MC_sample_size,quick_check_list)
    # # S_cp2 = Construct_Procedure_B(Evaluation, Aw,Au,fi, CL,T_m, param, MC_sample_size,quick_check_list)
    # # print("Construct_B real:",S_cp2[0],real_P_test(S_cp2[0],T_m, see, folder_name, sign))
    # minweight_S =  Solution_Initialization(Aw, param, -1)
    # print("============= 最小weight组合:",minweight_S)
    # middleweight_S = Solution_Initialization(Aw, param, -2)
    # print("============= 次小weight组合:",middleweight_S)
    # maxweight_S =  Solution_Initialization(Aw, param, -3)
    # print("============= 次次小weight组合:",maxweight_S)

    Tmax1 = [1*int(param[0])+2*x*int(param[0])/20 for x in range(20)]
    Tmax2 = [2*int(param[0])+3*x*int(param[0])/20 for x in range(20)]
    Tmax3 = [2*int(param[0])+3*x*int(param[0])/20 for x in range(20)]
    Tmax4 = [3*int(param[0])+3*x*int(param[0])/20 for x in range(20)]
    Tmax5 = [3*int(param[0])+3*x*int(param[0])/20 for x in range(20)]

    S1 =  Solution_Initialization(Aw, param, -1)
    S2 =  Solution_Initialization(Aw, param, -2)
    S3 =  Solution_Initialization(Aw, param, -3)
    S4 =  Solution_Initialization(Aw, param, -4)
    S5 =  Solution_Initialization(Aw, param, -5)

    r_p_list = []
    pool = Pool(processes = 5)
    r_p_list.append(pool.apply_async(real_P_test, (S1, Tmax1, see, IF, sign)))
    r_p_list.append(pool.apply_async(real_P_test, (S2, Tmax2, see, IF, sign)))
    r_p_list.append(pool.apply_async(real_P_test, (S3, Tmax3, see, IF, sign)))
    r_p_list.append(pool.apply_async(real_P_test, (S4, Tmax4, see, IF, sign)))
    r_p_list.append(pool.apply_async(real_P_test, (S5, Tmax5, see, IF, sign)))
    pool.close()
    pool.join()

    print("============= S1:",Tmax1)
    print("============= S1:",r_p_list[0].get())
    print("============= S2:",Tmax2)
    print("============= S2:",r_p_list[1].get())
    print("============= S3:",Tmax3)
    print("============= S3:",r_p_list[2].get())
    print("============= S4:",Tmax4)
    print("============= S4:",r_p_list[3].get())
    print("============= S5:",Tmax5)
    print("============= S5:",r_p_list[4].get())

    

    