'''
exp_Result_Greedy.py

Greedy正式实验

'''

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
from src.functions.func_solution_factor import Statistic_factor
from src.functions.func_greedy import Greedy
from src.functions.func_realP_statistical import real_P_test

def param_test(Aw, fi, param,T_m, CL, MC_sample_size, Evaluation, file_1, file_2, seed, quick_check_list, data_folder, sign, times):
    print(f"Greedy_{sign}_{int(param[0])}_{int(param[1])}_{int(param[2])}_{T_m} 第{times} 开始！")
    start_time = time.time()
    Solution_LS = Greedy(Evaluation, Aw, fi, CL, T_m, param, MC_sample_size, quick_check_list)
    running_time = time.time()-start_time
    real_CL = real_P_test(Solution_LS[0],T_m,seed, data_folder, sign)
    
    record_to_csv1(file_1,Solution_LS,running_time,real_CL,times)  
    if times == 0:
        record_to_csv2(file_2, Solution_LS[4])
    print(f"Greedy_{sign}_{int(param[0])}_{int(param[1])}_{int(param[2])}_{T_m} 第{times} 已完成！")

def create_csv(file_1, file_2):  
    path1 = my_file.real_path_of(f'result/Greedy/{file_1}')
    with open(path1,'w',newline='') as f:
        csv_write = csv.writer(f)
        csv_head = ['Times','Solution','Cost','Confidence Level','Evaluation Times','Real Confidence Level','Running Time']
        csv_write.writerow(csv_head)

    path2 = my_file.real_path_of(f'result/Greedy/{file_2}')
    with open(path2,'w',newline='') as f:
        csv_write = csv.writer(f)
        csv_head = ['Times','Cost','Confidence Level','Solution']
        csv_write.writerow(csv_head)

def record_to_csv1(file_1, Solution, running_t, real_CL, times):
    path1 = my_file.real_path_of(f'result/Greedy/{file_1}')
    with open(path1,'a',newline='') as f:
        csv_write = csv.writer(f)
        data_row = [times, Solution[0], Solution[1], Solution[2], Solution[3], real_CL, running_t]
        csv_write.writerow(data_row)

def record_to_csv2(file_2, Tem_Solution):
    path2 = my_file.real_path_of(f'result/Greedy/{file_2}')
    with open(path2,'a',newline='') as f:
        for tem_s in Tem_Solution:
            csv_write = csv.writer(f)
            data_row = [tem_s[3], tem_s[0], tem_s[1],tem_s[2]]
            csv_write.writerow(data_row)

if __name__=="__main__":

    se = 827323
    _lambda = 5
    Iterative_time = 30
    CL = 0.99
    sign  = ['lab','app']
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
                'benchmark/Instance_app_3_5_30_',
                'benchmark/Instance_app_4_5_30_',
                'benchmark/Instance_app_5_5_30_',
                'benchmark/Instance_app_5_10_30_',
                'benchmark/Instance_app_10_10_500_',
                'benchmark/Instance_app_10_20_500_',
                'benchmark/Instance_app_20_10_500_',
                'benchmark/Instance_app_30_10_500_',
                'benchmark/Instance_app_40_10_500_',
                'benchmark/Instance_app_50_10_500_']
            Tmax = [[21,27],[47,49],[27,38],[16,27],[32,37],[13,16],[43,48],[58,70],[85,95],[91,100]]


        
        for i in range(len(Instance_folder)):
            param = re.findall(r'\d+',Instance_folder[i])

            if int(param[2])**(int(param[0])) < 10**7:
                Evaluation =  advanced_exact_evaluation
            else:
                Evaluation =  advanced_monte_carlo

            MC_sample_size = 1000000
            quick_check_list = factorization_list(int(param[2]),int(param[0]),CL,10,20)

            for T_m in Tmax[i]:
                file_1 = f'Greedy_{s}_{int(param[0])}_{int(param[1])}_{int(param[2])}_{T_m}.csv'
                file_2 = f'Greedy_{s}_{int(param[0])}_{int(param[1])}_{int(param[2])}_{T_m}_temp.csv'
                create_csv(file_1,file_2)
                Aw,Au,Av,fi = Statistic_factor(param, _lambda, Instance_folder[i])

                for j in range(Iterative_time):
                    p = _Process(target=param_test,args=(Aw, fi, param,T_m,CL,MC_sample_size,Evaluation,file_1,file_2,se,quick_check_list, Instance_folder[i],s,j))
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

    print(f"{s}实验已结束！")
