'''
func_EDA.py

EDA算法函数

'''

import sys
sys.path.append(sys.path[0] + '/../')
import re
import heapq
import random
import numpy as np
from functions.func_solution_factor import solution_cost, confidence_list_evaluation, Solution_Initialization, Solution_Evaluation, Statistic_factor
from functions.func_cl_estimator import monte_carlo_estimate, advanced_monte_carlo, exact_evaluation, advanced_exact_evaluation,Hoeffding_estimate,Bernstein_estimate,factorization_list
from functions.func_realP_statistical import real_P_test

# 全局变量
PopulationSize = 10
EliteSize = 6
# 概率模型初始化
def ProbabilityInitialization(param):
    node_num, factor_num= int(param[0]), int(param[1])
    return [[1/factor_num for _ in range(factor_num)] for _ in range(node_num)]
# 更新概率模型
def UpdateProbability(Population,param):
    node_num, factor_num = int(param[0]), int(param[1])
    NewP = []    
    for node_index in range(node_num):
        count = [0 for _ in range(factor_num)]       
        for Single in Population:
            count[int((Single[node_index + 2]-node_index)/node_num)] += 1    # 序号计算公式：Single[node_index] = factor_index * node_num + node_index
        NewP.append([i/len(Population) for i in count])  
    return NewP   
# 按概率挑选个体
def random_pick(some_list, probabilities):
    x = random.uniform(0,1) 
    cumulative_probability = 0.0 
    for item, item_probability in zip(some_list, probabilities): 
        cumulative_probability += item_probability 
        if x < cumulative_probability:
            break    
    return item 
# 人口生成
def PopulationGeneration(Probability,param,node_resort_w):
    node_id_list = [i for i in range(int(param[0]))]   
    EDA_solution = np.zeros(len(node_id_list), dtype=int)   
    for node_id in node_id_list:
        factor_list = sorted(node_resort_w[node_id])
        EDA_solution[node_id] = random_pick(factor_list, Probability[node_id])    # 按每个环节内的概率向量随机生成解    
    return EDA_solution
# 修复模块
def RepairSolution(ConfidenceRecord, Eval_func, Solution, T_max, fi, node_resort_w, threshold_p, param, MC_sample_size, quick_check_list):
    _St = np.copy(Solution)
    available_S = np.copy(Solution)
    Eval = 0
    stopping_condition = -np.ones(len(available_S),int)
    select_node = -1

    # _pt, e = confidence_list_evaluation(ConfidenceRecord, Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)
    _pt = Solution_Evaluation(Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)
    Eval += 1

    if _pt < threshold_p:   # 如果为不可行解，进行修复
        while 1:
            factor_list = []
            for factor_id in _St:
                factor_list.append(fi[factor_id])

            solution_utility = [factor.utility for factor in factor_list]
            for i in range(len(available_S)):
                if available_S[i] == -1:
                    solution_utility[i] = np.inf

            select_node = np.argsort(solution_utility)[0] # 当前解中utility最小的factor所在的node
            select_factor = _St[select_node]
    
            if node_resort_w[select_node].index(select_factor) < len(node_resort_w[select_node])-1:     # 查询weight矩阵中比当前factor小的下一个因子
                _St[select_node] = node_resort_w[select_node][node_resort_w[select_node].index(select_factor)+1]          
                # _pt, e = confidence_list_evaluation(ConfidenceRecord, Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)  # 评估是否满足约束
                _pt = Solution_Evaluation(Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)
                Eval += 1
                if _pt >= threshold_p:
                    break
            else:  
                available_S[select_node] = -1   # 如果不再有更小的weight，随后的循环不再替换本次的select_node
                if (available_S == stopping_condition).all():
                    print("Warning: 没有找到可行解, 设为每个class下最小weight的item!")
                    _St = Solution_Initialization(node_resort_w, param, -1)
                    break  
        # _St = Solution_Initialization(node_resort_w, param, -1)
        # _pt, e = confidence_list_evaluation(ConfidenceRecord, Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)  # 评估是否满足约束
        # Eval += e 
    return solution_cost(_St,fi), _pt, list(np.copy(_St)), Eval
# EDA主函数
def EstimationDistributionAlgorithm(Eval_func,T_max, fi, node_resort_w, threshold_p, param,  EVAstoppingTimes, MC_sample_size, quick_check_list):
    #----------- 初始化confi_array -----------
    confi_array = np.ones(2+int(param[0]))
    S_init = Solution_Initialization(node_resort_w, param, -1)
    p_init = Solution_Evaluation(Eval_func, S_init, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)
    s_list = [1,p_init]
    s_list.extend(S_init)
    confi_array = np.vstack((confi_array,s_list))
    
    ProbabilityModel = ProbabilityInitialization(param)
    Eva = 0
    Tem_opt_solution = []

    while Eva < EVAstoppingTimes:
        #========= 生成种群 =================
        SolutionPopulation = []
        for _ in range(PopulationSize):
            InitSolutionSingle = PopulationGeneration(ProbabilityModel,param, node_resort_w)
            Fc,Fp,FeasibleSingle,e = RepairSolution(confi_array, Eval_func, InitSolutionSingle, T_max, fi, node_resort_w, threshold_p, param, MC_sample_size, quick_check_list)
            Eva += e
            FeasibleSingle.insert(0,Fc)
            FeasibleSingle.insert(1,Fp)
            heapq.heappush(SolutionPopulation, FeasibleSingle)  # 构建堆列表来存储最优个体
        
        #========== 选取优势个体 =============
        ElitePopulation = heapq.nsmallest(EliteSize,SolutionPopulation)
        Tem_opt_solution.append([ElitePopulation[0][0],ElitePopulation[0][1],ElitePopulation[0][2:],Eva])

        #========== 更新概率模型 =============
        ProbabilityModel = UpdateProbability(ElitePopulation,param)
        # print("EDA has been running:", Eva)
        # print(f"Iteration times: {t}")
    
    Optimal_Solution = ElitePopulation[0][2:]

    Optimal_p = ElitePopulation[0][1]
    Optimal_cost = ElitePopulation[0][0]

    return  Optimal_Solution, Optimal_cost, Optimal_p, Eva, Tem_opt_solution

if __name__=='__main__':
    '''
    评估方法一共有: 
    monte_carlo_estimate, advanced_monte_carlo, exact_evaluation, advanced_exact_evaluation, Hoeffding_estimate, Bernstein_estimate
    '''
    sel = 1
    MaxIter = 10000
    se = random.seed(827)
    if sel == 1:
        folder_name = 'benchmark/Instance_huawei_3_5_30_'
        param = re.findall(r'\d+',folder_name)
        T_m,threshold_p = 50,0.99
        Evaluation =  advanced_exact_evaluation
        MC_sample_size = 100000

        random.seed(233)
        for a in [2]:
            Aw,Au,Av,fi = Statistic_factor(param, a, folder_name)
            quick_check_list = factorization_list(int(param[2]),int(param[0]),threshold_p,10,20)
            Solution = EstimationDistributionAlgorithm(Evaluation,T_m, fi, Aw, threshold_p, param, MaxIter, MC_sample_size, quick_check_list)
            print("Solution:",Solution[0],"cost:",Solution[1],"confidence level:",Solution[2],"Evaluation times:",Solution[3])
    else:
        folder_name = 'benchmark/Instance_huawei_20_10_500_'
        param = re.findall(r'\d+',folder_name)
        T_m,threshold_p = 48,0.99
        Evaluation =  advanced_monte_carlo
        MC_sample_size = 100000

        random.seed(233)
        for a in [2]:
            Aw,Au,Av,fi = Statistic_factor(param, a, folder_name)
            quick_check_list = factorization_list(int(param[2]),int(param[0]),threshold_p,10,20)
            Solution = EstimationDistributionAlgorithm(Evaluation,T_m, fi, Aw, threshold_p, param, MaxIter,MC_sample_size, quick_check_list)
            print("EDA算法:",Solution[0],Solution[1],Solution[2],real_P_test(Solution[0], T_m, se, folder_name, 'huawei'),Solution[3])