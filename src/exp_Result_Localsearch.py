'''
exp_Result_Localsearch.py

正式实验

'''

import os
import sys
sys.path.append(sys.path[0] + '/../')
from src.utils import my_file
import re
import time
import csv
import numpy as np
from multiprocessing import Process as _Process
from typing import List
from src.functions.func_cl_estimator import advanced_exact_evaluation, advanced_monte_carlo, factorization_list
from src.functions.func_localsearch_cp import Construct_Procedure_B
from src.functions.func_solution_factor import Statistic_factor
from src.functions.func_localsearch import Local_Search_DD
from src.functions.func_realP_statistical import real_P_test

def param_test(Aw, Au, Av, fi, param,T_m, CL, MC_sample_size, Evaluation, _maxiter, file_0, file_1, file_2, file_3, file_record, seed, quick_check_list, data_folder, sign, times):
    np.random.randint(seed)
    print(f"LocalSearch_{sign}_{int(param[0])}_{int(param[1])}_{int(param[2])}_{T_m} 第{times} 开始！")
    start_time = time.time()
    S_cp = Construct_Procedure_B(Evaluation, Aw, Au, fi, CL, T_m, param, MC_sample_size,quick_check_list)
    Solution_LS = Local_Search_DD(Evaluation, S_cp, Av, fi, CL, T_m, param, _maxiter, MC_sample_size, quick_check_list)
    running_time = time.time()-start_time
    real_CL0 = real_P_test(Solution_LS[0][2], T_m, seed, data_folder, sign)
    real_CL1 = real_P_test(Solution_LS[1][2], T_m, seed, data_folder, sign)
    real_CL2 = real_P_test(Solution_LS[2][2], T_m, seed, data_folder, sign)
    EvaluationTimes = Solution_LS[4]
    record_to_csv1(file_0,Solution_LS[0],running_time,real_CL0,times,EvaluationTimes)
    record_to_csv1(file_1,Solution_LS[1],running_time,real_CL1,times,EvaluationTimes)
    record_to_csv1(file_2,Solution_LS[2],running_time,real_CL2,times,EvaluationTimes)
    if times == 0:
        record_to_csv2(file_3,Solution_LS[3],running_time, seed ,data_folder, sign, T_m)
        record_to_csv3(file_record, Solution_LS[5])
    print(f"LocalSearch_{sign}_{int(param[0])}_{int(param[1])}_{int(param[2])}_{T_m} 第{times} 已完成！")

def create_csv(file_0, file_1, file_2, file_3, file_record): 

    for df in [file_0, file_1, file_2]:
        path = my_file.real_path_of(f'result/Localsearch/{df}')
        with open(path,'w',newline='') as f:
            csv_write = csv.writer(f)
            csv_head = ['Times','Cost','Confidence Level','Evaluation Times','Solution','Real Confidence Level','Running Time']
            csv_write.writerow(csv_head)

    path = my_file.real_path_of(f'result/Localsearch/{file_3}')
    with open(path,'w',newline='') as f:
        csv_write = csv.writer(f)
        csv_head = ['Cost','Confidence Level','Solution','Real Confidence Level','Running Time']
        csv_write.writerow(csv_head)

    path = my_file.real_path_of(f'result/Localsearch/{file_record}')
    with open(path,'w',newline='') as f:
        csv_write = csv.writer(f)
        csv_head = ['Times','Cost','Confidence Level','Solution']
        csv_write.writerow(csv_head)

def record_to_csv1(file_, Solution, running_t, real_CL, times, EvaluationTimes):
    path = my_file.real_path_of(f'result/Localsearch/{file_}')
    with open(path,'a',newline='') as f:
        csv_write = csv.writer(f)
        data_row = [times, Solution[0], Solution[1], EvaluationTimes, Solution[2], real_CL, running_t]
        csv_write.writerow(data_row)

def record_to_csv2(file_, Solution_list, running_t, seed ,data_folder, ss, T):
    path = my_file.real_path_of(f'result/Localsearch/{file_}')
    for solution in Solution_list:
        real_CL = real_P_test(solution[2:], T, seed, data_folder, ss)
        with open(path,'a',newline='') as f:
            csv_write = csv.writer(f)
            data_row = [solution[0], solution[1], solution[2:], real_CL, running_t]
            csv_write.writerow(data_row)

def record_to_csv3(file_, Tem_Solution):
    path = my_file.real_path_of(f'result/Localsearch/{file_}')
    with open(path,'a',newline='') as f:
        for tem_s in Tem_Solution:
            csv_write = csv.writer(f)
            data_row = [tem_s[3], tem_s[0], tem_s[1],tem_s[2]]
            csv_write.writerow(data_row)

if __name__=="__main__":
    se = 832273
    _MAXITER = 30
    _lambda = 5
    Iterative_time = 30
    CL = 0.99
    sign  = ['lab','huawei']
    process_list: List[_Process] = []

    for s in sign:
        if s == 'lab':
            Instance_folder = [\
                'benchmark/Instance_lab_3_5_30_',
                'benchmark/Instance_lab_4_5_30_',
                'benchmark/Instance_lab_5_5_30_',
                'benchmark/Instance_lab_5_10_30_',
                'benchmark/Instance_lab_10_10_500_',
                'benchmark/Instance_lab_10_20_500_',
                'benchmark/Instance_lab_20_10_500_',
                'benchmark/Instance_lab_30_10_500_',
                'benchmark/Instance_lab_40_10_500_',
                'benchmark/Instance_lab_50_10_500_']  
            Tmax = [[11,14],[18,26],[11,20],[10,16],[19,23],[12,15],[25,32],[43,52],[55,63],[63,75]]

        else:
            Instance_folder = [\
                'benchmark/Instance_huawei_3_5_30_',
                'benchmark/Instance_huawei_4_5_30_',
                'benchmark/Instance_huawei_5_5_30_',
                'benchmark/Instance_huawei_5_10_30_',
                'benchmark/Instance_huawei_10_10_500_',
                'benchmark/Instance_huawei_10_20_500_',
                'benchmark/Instance_huawei_20_10_500_',
                'benchmark/Instance_huawei_30_10_500_',
                'benchmark/Instance_huawei_40_10_500_',
                'benchmark/Instance_huawei_50_10_500_']
            Tmax = [[21,27],[47,49],[27,38],[16,27],[32,37],[13,16],[43,48],[58,70],[85,95],[91,100]]

        
        for i in range(len(Instance_folder)):
            param = re.findall(r'\d+',Instance_folder[i])

            if int(param[2])**(int(param[0])) < 10**7:
                Evaluation =  advanced_exact_evaluation
            else:
                Evaluation =  advanced_monte_carlo

            MC_sample_size = 1000000
            quick_check_list = factorization_list(int(param[2]),int(param[0]),CL,10,50)   
            
            for T_m in Tmax[i]:
                file_0 = f'LocalSearch_{s}_{int(param[0])}_{int(param[1])}_{int(param[2])}_{T_m}_{0}.csv'
                file_1 = f'LocalSearch_{s}_{int(param[0])}_{int(param[1])}_{int(param[2])}_{T_m}_{1}.csv'
                file_2 = f'LocalSearch_{s}_{int(param[0])}_{int(param[1])}_{int(param[2])}_{T_m}_{2}.csv'
                file_3 = f'LocalSearch_{s}_{int(param[0])}_{int(param[1])}_{int(param[2])}_{T_m}_{3}.csv'
                file_record = f'LocalSearch_{s}_{int(param[0])}_{int(param[1])}_{int(param[2])}_{T_m}_record.csv'
                create_csv(file_0,file_1,file_2,file_3,file_record)
                Aw,Au,Av,fi = Statistic_factor(param, _lambda, Instance_folder[i])

                for j in range(Iterative_time):                                                       
                    p = _Process(target=param_test,args=(Aw,Au, Av, fi, param,T_m,CL,MC_sample_size,Evaluation,_MAXITER,file_0,file_1,file_2,file_3,file_record, se,quick_check_list, Instance_folder[i],s,j))
                    p.start()
                    process_list.append(p)

                    while len(process_list)>=30:
                        for p in process_list:
                            if not p.is_alive():
                                p.join()
                                process_list.remove(p)
                        if len(process_list)>=30:
                            time.sleep(1)

    for p in process_list:
        p.join()
    # 注意确认 Construct_Procedure.py 中样本数据的精确度
    print(f"{s}实验已结束！")