'''
func_GA.py

GA算法

'''

import sys
sys.path.append(sys.path[0] + '/../')
import re
import heapq
import random
import numpy as np
from src.utils import my_file
from src.functions.func_solution_factor import solution_cost, confidence_list_evaluation, Solution_Initialization, Solution_Evaluation, Statistic_factor
from src.functions.func_cl_estimator import monte_carlo_estimate, advanced_monte_carlo, exact_evaluation, advanced_exact_evaluation,Hoeffding_estimate,Bernstein_estimate,factorization_list
from src.functions.func_realP_statistical import real_P_test
# 全局变量

PopulationSize = 10 # 种群大小
EliteSize = 6       # 优势群体大小
pc = 0.1            # 交叉概率

# 种群初始化
def PopulationInitialization(ConfidenceRecord, Eval_func, T_max, fi, node_resort_w, threshold_p, param, MC_sample_size, quick_check_list, eva):
    node_num = int(param[0])
    InitPopulation = [] 
    for _ in range(PopulationSize):
        Single = []
        for node_index in range(node_num):
            Single.append(random.choice(node_resort_w[node_index]))
        _cost,_p,SingleRepair,e = RepairSolution(ConfidenceRecord, Eval_func, Single, T_max, fi, node_resort_w, threshold_p, param, MC_sample_size, quick_check_list)
        SingleRepair.insert(0,_cost)
        SingleRepair.insert(1,_p)
        heapq.heappush(InitPopulation,SingleRepair)
        eva += e
    return InitPopulation, eva
# 交叉算子
def CrossOver(ParentA,ParentB,param):
    node_num = int(param[0])
    Child = [-1 for _ in range(node_num)]
    Child[0] = ParentA[2]
    sign = 1
    for node_index in range(node_num):
        x = random.uniform(0,1)
        if x < pc:
            sign = sign*-1  # 如果满足条件，则对位点进行交叉

        if sign == 1:
            Child[node_index] = ParentA[node_index + 2]
        else:
            Child[node_index] = ParentB[node_index + 2]
    return Child
# 变异算子
def Mutation(Single, param, node_resort_w):
    node_num = int(param[0])
    pm = 1/node_num
    for node_index in range(node_num):
        factor_list = node_resort_w[node_index].copy()
        factor_list.remove(Single[node_index])
        x = random.uniform(0,1)
        if x < pm:
            Single[node_index] = random.choice(factor_list) # 如果满足条件，则对当前位点进行随机变异（除原因子）
    return Single
# 按概率挑选个体
def random_pick(some_list, probabilities):
    x = random.uniform(0,1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities): 
        cumulative_probability += item_probability 
        if x < cumulative_probability:
            break    
    return item, item_probability
# 优势个体挑选
def ParentPick(Population, Value):
    Probability = list(Value/sum(Value))
    ParentA, probA = random_pick(Population, Probability)    # 按每个环节内的概率向量随机选择个体A

    Population.remove(ParentA)
    Aindex = Probability.index(probA)
    Value.remove(Value[Aindex])
    Probability = Value/sum(Value)
    ParentB, _ = random_pick(Population, Probability)   # 按每个环节内的概率向量随机选择个体B
    return ParentA, ParentB
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
        _St = Solution_Initialization(node_resort_w, param, -1)
        # _pt, e = confidence_list_evaluation(ConfidenceRecord, Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)  # 评估是否满足约束
        _pt = Solution_Evaluation(Eval_func, _St, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)
        Eval += 1
    return solution_cost(_St,fi), _pt, list(np.copy(_St)), Eval
# GA主函数
def GeneticAlgorithm(Eval_func,T_max, fi, node_resort_w, threshold_p, param, EVAstoppingTimes, MC_sample_size, quick_check_list):
    #----------- 初始化confi_array -----------
    confi_array = np.ones(2+int(param[0]))
    S_init = Solution_Initialization(node_resort_w, param, -1)
    p_init = Solution_Evaluation(Eval_func, S_init, T_max, fi, threshold_p, param, MC_sample_size, quick_check_list)
    s_list = [1,p_init]
    s_list.extend(S_init)
    confi_array = np.vstack((confi_array,s_list))

    Tem_opt_solution = []
    Eva = 0
    SolutionPopulation, Eva = PopulationInitialization(confi_array, Eval_func, T_max, fi, node_resort_w, threshold_p, param, MC_sample_size, quick_check_list, Eva)
    while Eva < EVAstoppingTimes:
        #========== 选取优势个体作为父母种群 =============
        ElitePopulation = []
        EliteValue = []
        ElitePopulation = heapq.nsmallest(EliteSize,SolutionPopulation)
        EliteValue = [1/EP[0] for EP in ElitePopulation]

        for _ in range(int(EliteSize/2)):
            ParentA, ParentB = ParentPick(ElitePopulation,EliteValue)
            #========== 交叉，变异，修复 =============
            Child = CrossOver(ParentA, ParentB,param)
            Child = Mutation(Child,param,node_resort_w)
            cost,pt,Child,e = RepairSolution(confi_array, Eval_func, Child, T_max, fi, node_resort_w, threshold_p, param, MC_sample_size, quick_check_list)
            Eva += e
            Child.insert(0,cost)
            Child.insert(1,pt)
            heapq.heappush(SolutionPopulation,Child)

        print("GA has been running:", Eva)
        SolutionPopulation = heapq.nsmallest(PopulationSize,SolutionPopulation)
        Tem_opt_solution.append([SolutionPopulation[0][0],SolutionPopulation[0][1],SolutionPopulation[0][2:],Eva])

        heapq.heapify(SolutionPopulation)
        # print(f"Iteration times: {t}")
    
    St = heapq.heappop(SolutionPopulation)
    Optimal_Solution = np.copy(St[2:])
    Optimal_p = St[1]
    Optimal_cost = St[0]

    return  Optimal_Solution, Optimal_cost, Optimal_p ,Eva, Tem_opt_solution

if __name__=='__main__':
    '''
    评估方法一共有: 
    monte_carlo_estimate, advanced_monte_carlo, exact_evaluation, advanced_exact_evaluation, Hoeffding_estimate, Bernstein_estimate
    '''
    sel = 2
    MaxIter = 100
    se = random.seed(827)
    if sel == 1:
        folder_name = 'benchmark/Instance_huawei_3_4_100_'
        param = re.findall(r'\d+',folder_name)
        T_m,threshold_p = 50,0.99
        Evaluation =  advanced_exact_evaluation
        MC_sample_size = 100000

        random.seed(233)
        for a in [2]:
            Aw,Au,Av,fi = Statistic_factor(param, a, folder_name)
            quick_check_list = factorization_list(int(param[2]),int(param[0]),threshold_p,10,20)
            Solution = GeneticAlgorithm(Evaluation,T_m, fi, Aw, threshold_p, param, MaxIter,MC_sample_size, quick_check_list)
            print("Solution:",Solution[0],"cost:",Solution[1],"confidence level:",Solution[2],"Evaluation times:",Solution[3])
    else:
        folder_name = 'benchmark/Instance_huawei_10_5_200_'
        param = re.findall(r'\d+',folder_name)
        T_m,threshold_p = 65,0.99
        Evaluation =  advanced_monte_carlo
        MC_sample_size = 100000

        random.seed(233)
        for a in [2]:
            Aw,Au,Av,fi = Statistic_factor(param, a, folder_name)
            quick_check_list = factorization_list(int(param[2]),int(param[0]),threshold_p,10,20)
            print(Aw)
            Solution = GeneticAlgorithm(Evaluation,T_m, fi, Aw, threshold_p, param, MaxIter, MC_sample_size, quick_check_list)
            print("GA算法:",Solution[0],Solution[1],Solution[2],real_P_test(Solution[0], T_m, se, folder_name, 'huawei'),Solution[3])