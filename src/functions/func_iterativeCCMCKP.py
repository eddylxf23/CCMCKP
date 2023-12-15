import sys
sys.path.append(sys.path[0] + '/../')
sys.path.append('..')

import re
import math
from functions.func_Dyer_Zemel import Get_normal_mu_variance, Get_weight, Get_factor, Dynamic 
from functions.func_solution_factor import Solution_Evaluation, solution_cost
from functions.func_cl_estimator import monte_carlo_estimate
from scipy.stats import norm
from func_realP_statistical import real_P_test
import random
import copy
import time 

import func_timeout
from func_timeout import func_set_timeout
 
@func_set_timeout(3600) # 设置函数最大执行时间

def max_var(node_to_factors):

    node_num = len(node_to_factors)
    node_id_list = [i for i in range(node_num)]
    max_var_list = []
    for node_id in node_id_list:
        cur_factors = node_to_factors[node_id]
        max_var_list.append(max([f.sample_var for f in cur_factors]))

    return sum(max_var_list)


def CCMCKP_main(Evaluation, Tmax,  CL, param, MonteCarloTimes, QuickCheckList):
    _epsilon = 10**(-5)
    lambda_itera = 0
    Tmax_itera = Tmax
    C =  norm.ppf(CL)
    solution_set = []

    node_num = int(param[0])
    start_time = time.time()
    # k=1， 求解初始的RA-MCKP
    Value, n2f, f2o = Get_factor(param, folder_name)
    n2f_m, n2f_v = Get_normal_mu_variance(n2f)
    W = Get_weight(n2f, n2f_m, n2f_v, lambda_itera)
    sol = Dynamic(W, Value, n2f, Tmax_itera)
    p = Solution_Evaluation(Evaluation, sol, Tmax, f2o, CL, param, MonteCarloTimes, QuickCheckList)
    if p>=CL:
        solution_set.append(sol)

    iter = 0
    while p<CL:

        _sigma_2 = 0
        for j in range(len(n2f)):
            _sigma_2 += n2f_v[j][sol[j]//node_num]

        _sigma = math.sqrt(_sigma_2)
        lambda_itera = C/_sigma
        W = Get_weight(n2f, n2f_m, n2f_v, lambda_itera)
        sol = Dynamic(W, Value, n2f, Tmax_itera)
        p = Solution_Evaluation(Evaluation, sol, Tmax, f2o, CL, param, MonteCarloTimes, QuickCheckList)

        if p>=CL:
            solution_set.append(sol)

        iter += 1
        if iter >= 1000:
            print("error!")
            break

    maxvar = max_var(n2f)
    if lambda_itera == 0:
        print("\n初始解已满足, 全体解空间已被搜索。")
    else:
        point_a = [C**2/lambda_itera**2, Tmax_itera - C**2/lambda_itera]
        point_b = [maxvar, Tmax_itera - C*math.sqrt(maxvar)]
        a_x, a_y = point_a[0],point_a[1]
        b_x, b_y = point_b[0],point_b[1]
        lambda_ab = (b_y-a_y)/(b_x-a_x)
        T_ab = lambda_ab*(-a_x) + a_y

        # 记录各线段的斜率，截距和对应的两个端点
        Gamma_set = [lambda_ab]
        T_set = [T_ab]
        point_set = [[point_a,point_b]]

        while len(Gamma_set) !=0 :
            # 临时集合
            Gamma_set_tem = []
            T_set_tem = []
            point_set_tem = []

            # 计算各线段对应的解是否可行，若不可行就继续分割
            for i in range(len(Gamma_set)):
                lambda_itera = Gamma_set[i]
                Tmax_itera = T_set[i]
                W = Get_weight(n2f, n2f_m, n2f_v, lambda_itera)
                solu = Dynamic(W, Value, n2f, Tmax_itera)
                if solu is None:
                    continue

                p = Solution_Evaluation(Evaluation, solu, Tmax, f2o, CL, param, MonteCarloTimes, QuickCheckList)

                if p >= CL:
                    solution_set.append(solu)
                else:  
                    _sigma_2 = 0
                    _mu_ = 0
                    for j in range(len(n2f)):
                        _sigma_2 += n2f_v[j][solu[j]//node_num]
                        _mu_ += n2f_m[j][solu[j]//node_num]

                    point_tem = [_sigma_2, _mu_]
                    point_1 = point_set[i][0]
                    point_2 = point_set[i][1]
                    # point_1 = point_a
                    # point_2 = point_b

                    lambda_1 = (point_tem[1]-point_1[1])/(point_tem[0]-point_1[0]) - _epsilon
                    T_1 = lambda_1*(-point_1[0]) + point_1[1]

                    lambda_2 = (point_2[1]-point_tem[1])/(point_2[0]-point_tem[0]) + _epsilon
                    T_2 = lambda_2*(-point_2[0]) + point_2[1]

                    Gamma_set_tem.append(lambda_1)
                    Gamma_set_tem.append(lambda_2)
                    T_set_tem.append(T_1)
                    T_set_tem.append(T_2)
                    point_set_tem.append([point_1, point_tem])
                    point_set_tem.append([point_tem, point_2])
            
            if point_set == point_set_tem:
                break

            # Gamma_set = copy.deepcopy(Gamma_set_tem)
            # T_set = copy.deepcopy(T_set_tem)
            # point_set = copy.deepcopy(point_set_tem)
            Gamma_set = Gamma_set_tem
            T_set = T_set_tem
            point_set = point_set_tem

        
    obj_list = [solution_cost(s,f2o) for s in solution_set]
    if not obj_list:
        return None, [], time.time()-start_time

    optimal_obj = min(obj_list)

    optimal_solution = solution_set[obj_list.index(optimal_obj)]

    return optimal_obj, optimal_solution,time.time()-start_time

if __name__ == '__main__':
    
    try:
        se = random.seed(233)
        folder_name = 'benchmark/Instance_lab_4_5_30_'
        param = re.findall(r'\d+',folder_name)  
        CL = 0.9
        # samplesize = 30
        Tmax = 14
        Evaluation = monte_carlo_estimate
        MonteCarloTimes = 100000
        QuickCheckList = []

        optimal_obj, optimal_solution, tt =  CCMCKP_main(Evaluation, Tmax,  CL, param, MonteCarloTimes, QuickCheckList)
        Value, n2f, f2o = Get_factor(param, folder_name)
        print(Solution_Evaluation(Evaluation, optimal_solution, Tmax, f2o, CL, param, MonteCarloTimes, QuickCheckList))

        print("\n最优解为:", optimal_solution)
        print("最优成本为:", optimal_obj)
        print("运行时间为:", tt)
        # print(real_P_test(optimal_solution, Tmax, se, folder_name, 'lab'))

    except func_timeout.exceptions.FunctionTimedOut:
        print("超时了,自动退出")
                    


