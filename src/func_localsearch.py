'''
Local_search.py

Local_Swap_Search: 按照value去改善解
Degrade_:  解的退化
deblock: 解的解构
Local_Search_DD: 实现带着degrade和deblock模块的局部搜索

======================= 2022.04.01改动
1. localswapsearch 增加了随机生成一部分node_list的模块, 需要时调用;
2. localswapsearch 的遍历顺序改为倒序, furtherswapsearch不变;
3. localsearch 外层循环, 当退化次数达到Const时, 改为对当前最优解执行退化，而不是什么都不做;
4. localsearch 增加了堆记录队列, 方便以后如果需要提升最优解的可行概率;
'''

import sys
sys.path.append(sys.path[0] + '/../')
sys.path.append('..')

import re
import time
import heapq
import random
import numpy as np
from random import choice
from functions.func_cl_estimator import monte_carlo_estimate, advanced_monte_carlo, exact_evaluation, advanced_exact_evaluation,Hoeffding_estimate,Bernstein_estimate,factorization_list
from functions.func_localsearch_cp import Construct_Procedure_A,Construct_Procedure_B
from functions.func_solution_factor import solution_cost,Statistic_factor, confidence_list_evaluation, Solution_Evaluation
from functions.func_realP_statistical import real_P_test

_alpha = 0.7
UpConfidenceLevel = 0.995

def Local_Swap_Search(EvaluationFunction, Solution, p, node_resort_v, FactortoObject, CL, T_max,param, MonteCarloTimes, QuickCheckList, Eval_times,TemSolutionList,ConfidenceRecord, CostList,MonteCarloList):
    node_list = [i for i in range(int(param[0]))]
    _St,_S = np.copy(Solution), np.copy(Solution)
    _p = p
    random.shuffle(node_list)

    for node_id in node_list:
        select_factor = _St[node_id]
        # 遍历value矩阵中比factor大的所有因子
        index_ = node_resort_v[node_id].index(select_factor)

        for factor_id in node_resort_v[node_id][:index_][::-1]:
            _St[node_id] = factor_id
            # _pt, _t = confidence_list_evaluation(ConfidenceRecord, EvaluationFunction, _St, T_max, FactortoObject, CL, param, MonteCarloTimes, QuickCheckList)
            _pt = Solution_Evaluation(EvaluationFunction, _St, T_max, FactortoObject, CL, param, MonteCarloTimes, QuickCheckList)
            Eval_times += 1
            if _pt >= CL:
                _S = np.copy(_St)   # 尽可能找到最大value且满足置信度约束的那个因子
                _p = _pt
                _c = solution_cost(_S,FactortoObject)
                FeasibleSolutionList(_S,_c,_p,CostList,MonteCarloList)
                TemSolutionList.append([_c ,_p,_S,Eval_times])
                break               # node_resort_v从大到小排列，找到的第一个可行item即可跳出    
        _St = np.copy(_S)   # 更新_St

    feasible_Solution = _S
    obj_cost = solution_cost(_S,FactortoObject)
    p = _p
    return  obj_cost,p,feasible_Solution,Eval_times

def Degrade_(EvaluationFunction, Solution, _pp, node_resort_w ,FactortoObject, CL, T_max, param, MonteCarloTimes, ex_node, ex_factor, QuickCheckList, ConfidenceRecord):

    _St = np.copy(Solution)
    Eval_times = 0

    node_id = choice([i for i in range(int(param[0])) if i not in ex_node]) # 随机选择一个禁忌集外的node
    node_factor_ex =[_St[node_id]]

    _St[node_id] = choice([factor for factor in node_resort_w[node_id] if factor not in ex_factor])   # 随机选择一个非原因子的因子  
    node_factor_ex.append(_St[node_id]) # 更新当前node的禁忌集
    _pt = Solution_Evaluation(EvaluationFunction, _St, T_max, FactortoObject, CL, param, MonteCarloTimes, QuickCheckList)
    Eval_times += 1

    while _pt < CL:

        if len(node_factor_ex) < len(node_resort_w[node_id]):

            _St[node_id] = choice([factor for factor in node_resort_w[node_id] if factor not in node_factor_ex])   # 随机选择一个非原因子的因子  
            node_factor_ex.append(_St[node_id]) # 更新当前node的禁忌集
            # _pt, _t = confidence_list_evaluation(ConfidenceRecord, EvaluationFunction, _St, T_max, FactortoObject, CL, param, MonteCarloTimes, QuickCheckList)
            _pt = Solution_Evaluation(EvaluationFunction, _St, T_max, FactortoObject, CL, param, MonteCarloTimes, QuickCheckList)
            Eval_times += 1

        elif len(ex_node)<int(param[0])-1:
            # 该node除了原factor外所有其他factor都不可行，恢复原始解，并更换Node
            ex_node.append(node_id) # 更新node的禁忌集        
            _, _pt, _St, eval= Degrade_(EvaluationFunction, Solution, _pp, node_resort_w ,FactortoObject, CL, T_max, param,MonteCarloTimes,ex_node,ex_factor, QuickCheckList,ConfidenceRecord)
            Eval_times = eval

        else:
            # print("当前输入不能做任何的degrade!!!\n")
            _pt = _pp
            return solution_cost(Solution,FactortoObject),_pt,Solution,Eval_times

    # print(f"The Evaluation times of Degrade_ procedure is {Eval_times}")
    return solution_cost(_St,FactortoObject), _pt,_St, Eval_times

def Further_Swap_Search(EvaluationFunction, Solution, p, node_resort_v, FactortoObject, CL, T_max, param, MonteCarloTimes, QuickCheckList,Eval_times,TemSolutionList,ConfidenceRecord,CostList,MonteCarloList):

    _St = np.copy(Solution)
    _c = solution_cost(_St,FactortoObject)
    _p = p
    node_list = [i for i in range(int(param[0]))]
    random.shuffle(node_list)
    
    # 遍历所有Node对
    for i in range(int(param[0])-1):
        node_id1 = node_list[i]
        for j in range(i + 1,int(param[0])):
            node_id2 = node_list[j]
            solution_group = []
            # 遍历value矩阵中比原value大的因子组合
            for factor_id1 in node_resort_v[node_id1]:
                for factor_id2 in node_resort_v[node_id2]:
                    S_temp = np.copy(_St)
                    S_temp[node_id1], S_temp[node_id2]= factor_id1, factor_id2
                    c_st = solution_cost(S_temp,FactortoObject)
                    if c_st < _c:
                        # 构造堆
                        heapq.heappush(solution_group,list([c_st,S_temp]))

            while solution_group: 
                # 默认弹出最小值，因为取负，所以是比当前解cost小的最大值
                temp = heapq.heappop(solution_group)
                # _pt, e = confidence_list_evaluation(ConfidenceRecord, EvaluationFunction, temp[1], T_max, FactortoObject, CL, param, MonteCarloTimes, QuickCheckList)
                _pt = Solution_Evaluation(EvaluationFunction, temp[1], T_max, FactortoObject, CL, param, MonteCarloTimes, QuickCheckList)
                Eval_times += 1
                if _pt >= CL:
                    _St = np.copy(temp[1])   # 更新最优解
                    _p = _pt
                    _c = temp[0]
                    FeasibleSolutionList(_St,_c,_p,CostList,MonteCarloList)
                    TemSolutionList.append([_c,_p,_St,Eval_times])
                    break       # 不再继续评估当前class组合，开始下一组class

    feasible_Solution = np.copy(_St)
    obj_cost = solution_cost(_St,FactortoObject)
    p = _p
    # print(f"The Evaluation times of further swap search procedure is {Eval_times}")

    return  obj_cost, p, feasible_Solution, Eval_times

def FeasibleSolutionList(Solution,cost,ConfidenceLevel,CostQueue,MonteCarloQueue):
    def push_Cost(queue,Solution,cost):
        # queue格式：[(cost,factor,...),(cost,factor,...),...]
        Queue_size = 30
        Solution.insert(0,-cost)    # heap 默认弹出最小value, 取负
        if Solution not in queue:
            if len(queue) < Queue_size:
                    heapq.heappush(queue, Solution)
            else:
                heapq.heappushpop(queue, Solution)

    def push_MonteCarlo(queue,Solution,cost,ConfidenceLevel):
        # queue格式：[tuple(MC_p,cost,factor,...),tuple(MC_p,cost,factor,...),...]
        Queue_size = 30
        cost_solu_list = []

        for q in queue:
            cost_solu_list.append(q[1:])
        Solution.insert(0,cost)

        if tuple(Solution) not in cost_solu_list:       #判断是否已在队列中
            Solution.insert(0,ConfidenceLevel)
            queue.append(tuple(Solution))
            if len(queue) > Queue_size:                     #队列溢出，剔除一个
                queue.sort(key=lambda x:(-x[0],x[1]))
                queue.pop()
        else:
            pass

    push_MonteCarlo(MonteCarloQueue,list(np.copy(Solution)),cost,ConfidenceLevel)
    push_Cost(CostQueue,list(np.copy(Solution)),cost)

def UpdateFeasibleSolutionList(Clist,Mlist,ConfidenceRecordList, EvaluationFunction, T_max, FactortoObject, ConfidenceLevel, param, MonteCarloTimes, QuickCheckList, Eval_t):
    New_CostList = heapq.nlargest(len(Clist),Clist)
    for i in range(len(New_CostList)):
        New_CostList[i][0] = -New_CostList[i][0]
        for _ in range(2):
            # _p, e = confidence_list_evaluation(ConfidenceRecordList, EvaluationFunction, New_CostList[i][1:], T_max, FactortoObject, ConfidenceLevel, param, MonteCarloTimes, QuickCheckList)
            _p = Solution_Evaluation(EvaluationFunction, New_CostList[i][1:], T_max, FactortoObject, ConfidenceLevel, param, MonteCarloTimes, QuickCheckList)
            Eval_t += 1
        New_CostList[i].insert(1,_p)

    New_MonteCarloList = heapq.nlargest(len(Mlist),Mlist)
    New_MonteCarloList = [list(NM) for NM in New_MonteCarloList]
    for j in range(len(New_MonteCarloList)):
        for _ in range(2):
            # _p, e = confidence_list_evaluation(ConfidenceRecordList, EvaluationFunction, New_MonteCarloList[j][2:], T_max, FactortoObject, ConfidenceLevel, param, MonteCarloTimes, QuickCheckList)
            _p = Solution_Evaluation(EvaluationFunction, New_MonteCarloList[j][2:], T_max, FactortoObject, ConfidenceLevel, param, MonteCarloTimes, QuickCheckList)
            Eval_t += 1
        New_MonteCarloList[i][0] = _p
    New_MonteCarloList.sort(reverse=True)   # 按评估的置信度从大到小排列

    return New_CostList, New_MonteCarloList, Eval_t

def Local_Search_DD(EvaluationFunction, S_cp, node_resort_v, FactortoObject, ConfidenceLevel, T_max, param, MaxIter, MonteCarloTimes, QuickCheckList):
    S_opt,_St = np.copy(S_cp[0]),np.copy(S_cp[0])
    c_opt,_ct = S_cp[1],S_cp[1]
    _p,_pt = S_cp[2],S_cp[2]
    t = 0
    Eval_times = S_cp[3]
    Tem_solution = S_cp[4]
    ConfidenceRecordList = np.ones(2+int(param[0]))
    MonteCarloList = []
    CostList = []
    #----------- 初始化ConfidenceRecordList, MonteCarloList, CostList -----------
    s_list = [1,_p]
    s_list.extend(S_opt)
    ConfidenceRecordList = np.vstack((ConfidenceRecordList,s_list))
    FeasibleSolutionList(S_opt,c_opt,_p,CostList,MonteCarloList)

    while t < MaxIter:
        _ct,_pt,_St,Eval_times = Local_Swap_Search(EvaluationFunction, _St, _pt, node_resort_v, FactortoObject, ConfidenceLevel, T_max, param, MonteCarloTimes, QuickCheckList,Eval_times,Tem_solution,ConfidenceRecordList, CostList,MonteCarloList)
        # print(f"Local_Swap_Search: {[_St,_ct,_pt,eva]}")
        if _ct < c_opt:
            S_opt = np.copy(_St)
            c_opt = _ct
            _p = _pt
            FeasibleSolutionList(S_opt,c_opt,_p,CostList,MonteCarloList)
        ex_node=[]
        ex_factor = [factor for factor in _St] # 禁忌集初始化
        _ct,_pt,_St,eva = Degrade_(EvaluationFunction, _St, _pt, node_resort_v,FactortoObject, ConfidenceLevel, T_max, param, MonteCarloTimes, ex_node, ex_factor, QuickCheckList, ConfidenceRecordList)
        FeasibleSolutionList(_St,_ct,_pt,CostList,MonteCarloList)
        Eval_times += eva
        t += 1

    c_opt,_p,S_opt,Eval_times = Further_Swap_Search(EvaluationFunction, S_opt, _p, node_resort_v, FactortoObject, ConfidenceLevel, T_max, param, MonteCarloTimes, QuickCheckList,Eval_times,Tem_solution,ConfidenceRecordList, CostList,MonteCarloList)
    # print(f"-------------- Optimal: {S_opt},{c_opt},{_p}")
    FeasibleSolutionList(S_opt,c_opt,_p,CostList,MonteCarloList)
    CostList,MonteCarloList,Eval_times = UpdateFeasibleSolutionList(CostList,MonteCarloList,ConfidenceRecordList, EvaluationFunction, T_max, FactortoObject, ConfidenceLevel, param, MonteCarloTimes, QuickCheckList, Eval_times)
    
    # 原始输出：最小cost
    Optimal_Solution = CostList[0][2:]
    Optimal_cost = CostList[0][0]
    Optimal_p = CostList[0][1]
    Optimal_0 = [Optimal_cost,Optimal_p,Optimal_Solution]

    # 变体1：以0.995为置信度阈值输出
    maxP = 0
    for CostSolution in CostList:
        if CostSolution[1] >= maxP:
            sub_cost = CostSolution[0]
            maxP = CostSolution[1]
            sub_Solution = CostSolution[2:]
                 
        if CostSolution[1] >= UpConfidenceLevel:       # 若遇到置信度大于0.995的解 
            Optimal_cost = CostSolution[0]
            maxP = CostSolution[1]
            Optimal_p = CostSolution[1]
            Optimal_Solution = CostSolution[2:]
            break
        
    if maxP < UpConfidenceLevel:  # 若解集没有符合条件的解，则取遇到的最大置信度的为输出解
        Optimal_Solution = sub_Solution
        Optimal_cost = sub_cost
        Optimal_p = maxP
    
    Optimal_1 = [Optimal_cost,Optimal_p,Optimal_Solution]
    
    # 变体3： 输出Cost解集中的前十个
    Optimal_3 = CostList[:10]

    # 变体2： 两个解集，加权输出
    CostOnlySolution = [s[2:] for s in CostList]
    MonteCarloOnlySolution = [list(s[2:]) for s in MonteCarloList]
    TS = CostOnlySolution + MonteCarloOnlySolution # 合并两个集合的解
    TotalSolution = []
    [TotalSolution.append(x) for x in TS if x not in TotalSolution]    # 去除重复元素
    TotalScore = [0 for _ in range(len(TotalSolution))]
    WholeSolution = []

    for s in range(len(TotalSolution)):
        CostScore = 0
        MonteCarloScore = 0
        if TotalSolution[s] in CostOnlySolution:
            _index =  CostOnlySolution.index(TotalSolution[s])
            CostScore = 30 - _index
            _p = CostList[_index][1]
            _c = CostList[_index][0]
        if TotalSolution[s] in MonteCarloOnlySolution:
            _index =  MonteCarloOnlySolution.index(TotalSolution[s])
            MonteCarloScore = 30 -_index
            _p = MonteCarloList[_index][0]
            _c = MonteCarloList[_index][1]
        TotalScore[s] = _alpha * CostScore + (1-_alpha) * MonteCarloScore
        WholeSolution.append([TotalScore[s], _c, _p] + TotalSolution[s])

    WholeSolution.sort()
    Optimal_Solution = WholeSolution[-1][3:]
    Optimal_cost = WholeSolution[-1][1]
    Optimal_p = WholeSolution[-1][2]
    Optimal_2 = [Optimal_cost,Optimal_p,Optimal_Solution]
    # Optimal_2 =[WholeSolution1[-1][1:],WholeSolution2[-1][1:],WholeSolution3[-1][1:],WholeSolution4[-1][1:]]

    return  Optimal_0, Optimal_1, Optimal_2, Optimal_3, Eval_times,Tem_solution

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

    folder_name = 'benchmark/Instance_lab_10_10_500_'
    param = re.findall(r'\d+',folder_name)
    Tmax = 19
    CL = 0.99
    _lambda = 2
    # Evaluation =  advanced_exact_evaluation
    Evaluation = monte_carlo_estimate
    MaxIter_LS = 30
    MonteCarloTimes = 100000 
    #samplesize_list = [2,5,10,20,50,100,500,1000]
    samplesize = int(param[2])

    se = random.seed(233)

    start = time.time()
    Aw,Au,Av,FactortoObject = Statistic_factor(param,  _lambda, folder_name)

    # QuickCheckList = factorization_list(samplesize,int(param[0]),CL,10,20)
    QuickCheckList = []
    S_cp2 = Construct_Procedure_B(Evaluation, Aw,Au,FactortoObject, CL, Tmax, param, MonteCarloTimes,QuickCheckList)
    Solution2 = Local_Search_DD(Evaluation,S_cp2, Av, FactortoObject, CL, Tmax,param,MaxIter_LS, MonteCarloTimes,QuickCheckList)
    # solu = [Av[ia][0] for ia in range(len(Av))]
    # print("Solution(mini):",solu, solution_cost(solu,FactortoObject))

    print("***************** TIME: ", time.time()-start)
    print("Solution(minicost):",Solution2[0])
    print("Solution(95):",Solution2[1])
    print("Solution(weighted):",Solution2[2])
    print("============================================")
    
    # print(real_P_test(Solution2[0][2], T_m, se, folder_name, 'lab'))
    # print(real_P_test(solu, Tmax, se, folder_name, 'lab'))
    # print(real_P_test(Solution2[0][2], Tmax, se, folder_name, 'lab'))


