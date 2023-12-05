'''
评估方法及不等式估计
'''
import sys
sys.path.append('.')
import math
import time
import heapq
import random
from random import choice
import itertools
import numpy as np
from src.utils.factor import Factor
from scipy.optimize import fsolve

def factorization_list(L,m,P0,list_size,K):
    #============= 计算乘积 =============
    def list_product(list_,size):
        if size == 0:
            return 1    
        else:
            return list_[size-1] * list_product(list_,size - 1)

    #============= 质因数分解 =============
    def factorization(n, m=2, result_list=[], count_=0):
        
        for i in range(m,int(n**0.5)+1):
            if n%i==0 and i!=1:
                result_list.append(i)
                count_=1
                break
        if count_==0:
            result_list.append(n)
            return result_list
        # print(int(n // i))
        return factorization(int(n//i),m=i,result_list=result_list)

    #============= 不进行全排列，随机生成 =============
    def permuta_list(arglist,K):
        length = len(arglist)
        if length == 1:
            return arglist[0]
        
        res_list = []
        for _ in range(K):
            index_list = list(np.copy(arglist))
            res = []
            for _ in range(length):
                a = choice(index_list)
                res.append(a)
                index_list.remove(a)
            res_list.append(res)

        return res_list

    #============= 均匀分m份 ===============
    def divideIntoNstrand(listTemp, n):
        twoList = [[] for i in range(n)]
        for i,e in enumerate(listTemp):
            twoList[i%n].append(e)
        return twoList

    #============= 主程序 ==============
    # L 应为10或100的整数倍
    if L >= 100:
        L_ = L // 100
        num_L = int(math.pow(L_,m))  #避免小数点精度问题
        num_b = int(math.pow(10,m)/round(1/(1-P0)))
        num_ = int(math.pow(10,m))
        f_list = factorization(num_L,result_list=[]) + factorization(num_b,result_list=[]) + factorization(num_,result_list=[])
        
    else:
        L_ = L // 10
        num_L = int(math.pow(L_,m))  #避免小数点精度问题
        num_b = int(math.pow(10,m)/round(1/(1-P0)))
        f_list = factorization(num_L,result_list=[]) + factorization(num_b,result_list=[])

    # print('分解数为:',L**m,'因子为：',f_list)
    if max(f_list)>L:
        print("因子分解失败, 质因子最大值为:", max(f_list))
        a = []
        return a
    
    product_list = []
    iter = 0
    while len(product_list) < list_size and iter < 1000:
        iter += 1
        product_schedule = [L for _ in range(m)]
        while max(product_schedule) >= L:
            np.random.shuffle(f_list)
            sliced_schedule = divideIntoNstrand(f_list,m)
            product_schedule = sorted([list_product(i,len(i))-1 for i in sliced_schedule])
       
        _index = np.where(np.all(product_schedule==product_list))[0]
        if _index.size > 0:
            continue
        else:  
            product_list.append(product_schedule)
            # print('划分为：', sliced_schedule)

    res1 = []# 去重
    for i in product_list:
        if i not in res1:
            res1.append(i)
    # print(res1)
    res2 = []# 全排列且去重
    for i in res1:
        res3 = permuta_list(i,K)
        for j in res3:
            if j not in res2:
                res2.append(j)

    return res2

def _monte_carlo_evaluate(samples, T_max):
    '''
    :param samples: 1-D array of samples
    :param upper_value:
    :return:
    '''

    return len(np.where(samples < T_max)[0]) / len(samples)

def monte_carlo_estimate(factor_list, T_max, MC_sample_size, CL, param, quick_check_list):
    '''
    Monte Carlo Estimation
    :param factor_list: solution, list of factors
    :param sample_size: monte carlo sample size
    :return: estimated Condence Level P
    '''
    sum_samples = np.zeros(MC_sample_size)

    for factor in factor_list:
        cur_samples = factor.resample_multi(MC_sample_size)
        sum_samples += cur_samples

    return _monte_carlo_evaluate(sum_samples, T_max)

def advanced_monte_carlo(factor_list, T_max, MC_sample_size, CL, param, quick_check_list):
    L,m = int(param[2]),int(param[0])
    _prob = 1- CL
    sign = 0
    _data_l = math.ceil(round(L**m * _prob)**(1/m))
    data_sum = np.sum([factor.get_index_sample(_data_l) for factor in factor_list])
    if data_sum >= T_max:
        # print(data_sum)
        sign = 1

    for i in range(len(quick_check_list)):
        data_sum = np.sum([factor_list[j].get_index_sample(quick_check_list[i][j]) for j in range(m)])
        # print(data_sum)
        if data_sum >= T_max:
            sign = 1

    if sign == 0:
        sum_samples = np.zeros(MC_sample_size)

        for factor in factor_list:
            cur_samples = factor.resample_multi(MC_sample_size)
            sum_samples += cur_samples
        
        return _monte_carlo_evaluate(sum_samples,T_max)
    
    else:
        return 0 # 直接判定为不可行解

def exact_evaluation(factor_list, T_max, MC_sample_size, CL, param, quick_check_list):
    '''
    exact_evaluation: 采用堆排序的方法

    factor: Factor类item因子组成的待定解
    T_max: 时延上界
    CL: 置信度下界
    param: benchmark系数,[class数,item数,样本数,0,整数的CL]

    output: 因子list是否满足置信度的精确判断
    '''
    L,m = int(param[2]),int(param[0])
    _prob = 1 - CL
    queue = []
    samples_combination = []

    def push(D):    # D is a m-dimension list index of samples
        if all([v < L for v in D]):
            samples_sum = np.sum([factor_list[i].get_index_sample(D[i]) for i in range(m)])
            D.insert(0,-samples_sum)    # heap 默认弹出最小值，此处取负，实为最大值
            if D not in queue:
                heapq.heappush(queue,D)

    push([0 for _ in range(m)])
    max_eval = math.ceil(L**m * _prob)
    pop_factor_sample_sum = T_max + 1
    
    while queue and pop_factor_sample_sum > T_max:
        pop_solution = heapq.heappop(queue)
        samples_combination.append(pop_solution)
        D = pop_solution[1:]   # 取出pop出数据组合的Index部分
        pop_factor_sample_sum = -pop_solution[0]    # 取出pop出数据组合的总时延部分

        for i in range(m):
            if D[i]<L-1:
                # 寻找pop出队列的数据组合的邻居，push入队列中
                d = list(np.copy(D))
                d[i] += 1
                push(d)

        # if len(samples_combination) > max_eval:
        #     pop_solution[0] = -pop_solution[0]
        #     print("不可行解，分界点的解为：",pop_solution)
        #     return 0    # 直接判定为不可行解
    
    return 1-(len(samples_combination)/L**m)

def advanced_exact_evaluation(factor_list, T_max, MC_sample_size, CL, param, quick_check_list):
    '''
    advanced_exact_evaluation: 采用堆排序的方法，增加了快筛过程，并且降低了迭代的次数

    factor: Factor类item因子组成的待定解
    T_max: 时延上界
    CL: 置信度下界
    param: benchmark系数,[class数,item数,样本数,0,整数的CL]

    output: 因子list是否满足置信度的精确判断
    '''
    L,m = int(param[2]),int(param[0])
    _prob = 1 - CL
    _prob = float('{:.3f}'.format(_prob)) # 避免精度问题
    # k = math.ceil(L**m * _prob)
    queue = []
    samples_combination = []

    def push_q(D):    # D is a m-dimension list index of samples
        if all([v < L for v in D]):
            samples_sum = np.sum([factor_list[i].get_index_sample(D[i]) for i in range(m)])
            D.insert(0,-samples_sum)    # heap 默认弹出最小值，此处取负，实为最大值
            if D not in queue:
                heapq.heappush(queue,D)

    max_eval = math.ceil(L**m * _prob)
    _data_lz = math.ceil((L**m * _prob)**(1/m))
    sign = 0
    data_sum = np.sum([factor.get_index_sample(_data_lz) for factor in factor_list])

    if data_sum >= T_max:
        # print(data_sum)
        sign = 1

    for i in range(len(quick_check_list)):
        data_sum = np.sum([factor_list[j].get_index_sample(quick_check_list[i][j]) for j in range(m)])
        # print(data_sum)
        if data_sum >= T_max:
            sign = 1

    if sign == 0:
        
        while data_sum < T_max:
            if _data_lz > 0:
                _data_lz -= 1
                data_sum = np.sum([factor.get_index_sample(_data_lz) for factor in factor_list])
            else: 
                break
        
        select_pool = range(_data_lz + 2)   # 数据序号从[0,1,2,...,_data_lz,_data_lz+1]
        #  迭代生成数据组合
        for selected_cand_idx in itertools.product(select_pool, repeat = m):
            can_sample_list = list(selected_cand_idx)

            if _data_lz + 1 in can_sample_list:
                push_q(can_sample_list)    # 任何包含_data_lz + 1的数据组合都Push入队列中

        pop_factor_sample_sum = T_max + 1
        
        while queue and pop_factor_sample_sum > T_max:
            pop_solution = heapq.heappop(queue)
            samples_combination.append(pop_solution)

            for i in range(m):
                D = pop_solution[1:]   # 取出pop出数据组合的Index部分
                pop_factor_sample_sum = -pop_solution[0]    # 取出pop出数据组合的总时延部分

                if D[i]<L-1:
                    # 寻找pop出队列的数据组合的邻居，push入队列中
                    D[i] += 1
                    push_q(D)

            if len(samples_combination)>max_eval:
                return 0 # 直接判定为不可行解
        
        return 1-((len(samples_combination)+(_data_lz + 1)**m)/L**m)
    
    else:
        return 0 # 直接判定为不可行解

def Hoeffding_estimate(factor_list, T_max, MC_sample_size, CL, param, quick_check_list):
    '''
    Hoeffding
    :param factor_list: solution, which is a list of Factors
    :param upper_bound: 时延上界
    :return: estimated P
    '''
    max_K = 10
    # calculate D
    D = 0
    for factor in factor_list:
        D += (factor.moments_min[2] / factor.moments_min[1]) ** 2

    # calculate t
    tmp1 = [(factor.moments_min[1] + factor.sample_min) for factor in factor_list]
    t = T_max - np.sum(tmp1)

    if t < 0:
        # print('Hoeffding Error: t should be positive')
        # print(f't={upper_bound}-{np.sum(tmp1)}')
        return 0

    sum_all = 0

    for factor in factor_list:
        c_ = factor.sample_max - factor.sample_min
        y_ = 4 * t * c_ / D

        PV_1 = c_ ** (max_K - 1) * factor.moments_min[1] / factor.moments_min[max_K]   # ϕ
        PV_2 = 1 # w

        sum_1 = 0
        sum_2 = PV_1 - 1

        for k in range(1, max_K - 2 + 1):
            PV_1 *= factor.moments_min[k + 1] / (c_ * factor.moments_min[k])
            sum_1 += PV_2 * (PV_1 - 1)

            PV_2 *= y_ / k
            sum_2 += PV_2 * (PV_1 - 1)

        expy = np.exp(y_)
        Cp_ = ((expy + sum_1) / (expy + sum_2)) ** 2

        sum_all += c_ ** 2 * Cp_

    result_prob = 1 - np.exp(- 2 * t ** 2 / sum_all)

    return result_prob

def Bernstein_estimate(factor_list, T_max, MC_sample_size, CL, param, quick_check_list):
    '''
    Bernstein
    :param factor_list: solution, which is a list of Factors
    :param upper_bound: 时延上界
    :return: estimated P
    '''

    c_list = []
    first_moments = []

    for factor in factor_list:
        c_list.append(factor.sample_max - factor.sample_min)
        first_moments.append(factor.moments_origin[1])

    #
    C = np.max(c_list)
    t = T_max - np.sum(first_moments)

    if t < 0:
        # print('Bernstein Error: t should be positive')
        return 0

    V = 0
    for factor in factor_list:
        tmp1 = (factor.samples - factor.moments_origin[1]) ** 2
        V += np.mean(tmp1)

    P = 1 - np.exp(- (t ** 2 / 2) / (V + C * t / 3))

    return P


# def factorization_list(L,m,P0,list_size,K):
#     # 计算乘积
#     def list_product(list_,size):
#         if size == 0:
#             return 1    
#         else:
#             return list_[size-1] * list_product(list_,size - 1)

#     # 质因数分解
#     def factorization(n, m=2, a=[], count_=0):
        
#         for i in range(m,int(n**0.5)+1):
#             if n%i==0 and i!=1:
#                 a.append(i)
#                 count_=1
#                 break
#         if count_==0:
#             a.append(n)
#             return a
#         return factorization(int(n/i),m=i,a=a)

#     # 不进行全排列，随机生成
#     def permuta_list(arglist,K):
#         length = len(arglist)
#         if length == 1:
#             return arglist[0]
        
#         res_list = []
#         for _ in range(K):
#             index_list = list(np.copy(arglist))
#             res = []
#             for _ in range(length):
#                 a = choice(index_list)
#                 res.append(a)
#                 index_list.remove(a)
#             res_list.append(res)

#         return res_list

#     # 尽量均匀分m份
#     def divideIntoNstrand(listTemp, n):
#         twoList = [[] for i in range(n)]
#         for i,e in enumerate(listTemp):
#             twoList[i%n].append(e)
#         return twoList

#     num = L**m/round(1/(1-P0))  #避免小数点精度问题
            
#     t1=time.time()
#     f_list = factorization(num)
#     print('分解数为:',num,'因子为：',f_list)
#     if max(f_list)>L:
#         print("因子分解失败, 质因子最大值为:", max(f_list))
#         exit()
    
#     product_list = []
#     iter = 0
#     while len(product_list) < list_size and iter < 1000:
#         iter += 1
#         product_schedule = [L for _ in range(m)]
#         while max(product_schedule) >= L:
#             np.random.shuffle(f_list)
#             sliced_schedule = divideIntoNstrand(f_list,m)
#             product_schedule = sorted([list_product(i,len(i))-1 for i in sliced_schedule])
       
#         _index = np.where(np.all(product_schedule==product_list))[0]
#         if _index.size > 0:
#             continue
#         else:  
#             product_list.append(product_schedule)
#             # print('划分为：', sliced_schedule)

#     res1 = []# 去重
#     for i in product_list:
#         if i not in res1:
#             res1.append(i)
#     print(res1)
#     res2 = []# 全排列且去重
#     for i in res1:
#         res3 = permuta_list(i,K)
#         for j in res3:
#             if j not in res2:
#                 res2.append(j)
#     t2=time.time()
#     print('耗时：',t2-t1)

#     return res2
    
if __name__ =='__main__':

    L = 400
    m = 13
    P0 = 0.99999
    print(len(factorization_list(L,m,P0,10,10)))
