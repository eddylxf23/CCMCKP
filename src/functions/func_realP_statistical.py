'''
真实置信度评估 和 结果统计汇总
'''

import sys
sys.path.append(sys.path[0] + '/../')
sys.path.append("..")
import numpy as np
import pandas as pd
from utils import my_file
from multiprocessing import Pool
from src.benchmark.benchmark_lab import Gamma_, Uniform, Fatiguelife, Bimodal, Truncated_norm
from src.benchmark.benchmark_huawei import Gamma_h, Uniform_h, Fatiguelife_h, Bimodal_h, Truncated_norm_h

def real_P_test(solution, T_max, see, data_folder, sign):

    np.random.seed(see)
    final_s = np.copy(solution) 
    big_sample_size = 1*10**7
    sum_samples = np.zeros(big_sample_size)
    prob_,ind_ = [0.9,0.09,0.009,0.001],[0,10,20,30]

    for factor_id in final_s:
        filename_ = f'{factor_id}_dist.txt'
        with open(my_file.real_path_of(data_folder, filename_), 'r') as f:
            data = f.readlines()
        distribution = data[0].strip().split()[1]
        mu_ = float(data[1].strip().split()[1])
        variance_ = float(data[2].strip().split()[1])

        if sign == 'huawei':
            #-------------- Huawei 长尾数据 --------------
            if distribution == 'uniform':
                sum_samples += Uniform_h(big_sample_size,ind_,prob_)
            elif distribution == 'fatiguelife':
                sum_samples += Fatiguelife_h(big_sample_size,ind_,prob_,mu_,variance_)
            elif distribution == 'bimodal':
                select = int(data[5].strip().split()[1])
                sum_samples += Bimodal_h(big_sample_size,ind_,prob_,mu_,variance_,select)
            elif distribution == 'gamma':
                sum_samples += Gamma_h(big_sample_size,mu_,variance_)
            else:
                sum_samples += Truncated_norm_h(big_sample_size,ind_,prob_,mu_,variance_)
        elif sign == 'lab':
            #-------------- Lab 数据 --------------
            if distribution == 'uniform':
                sum_samples += Uniform(big_sample_size,mu_)
            elif distribution == 'fatiguelife':
                sum_samples += Fatiguelife(big_sample_size,mu_,variance_)
            elif distribution == 'bimodal':
                select = int(data[5].strip().split()[1])
                sum_samples += Bimodal(big_sample_size,mu_,variance_,select)
            elif distribution == 'gamma':
                sum_samples += Gamma_(big_sample_size,mu_,variance_)
            else:
                sum_samples += Truncated_norm(big_sample_size,mu_,variance_)
        else:
            #-------------- Huawei extre 长尾数据 --------------
            h = 0

    if isinstance(T_max,list):
        smaller_num = []
        for tm in T_max:
            smaller_num.append(len(np.where(sum_samples <= tm)[0]))
        print(final_s,"多个tmax的真实值完成")
        return tuple(sn / big_sample_size for sn in smaller_num)
    else:
        smaller_num = len(np.where(sum_samples <= T_max)[0])
        print(final_s,"真实值完成")
        return smaller_num / big_sample_size


def rearrage_result(data_c, data_MC, _gamma, T_max, file_, file_2, file_3, file_4, data_folder, _P, seed, sign):

    see = np.random.seed(seed)
    result_folder = 'out/'

    #--------------- 评估真实值 ----------------
    r_p_list = []
    real_p_list = []
    pool = Pool(processes = 50)
    for i in range(data_c.shape[0]):
        r_p_list.append(pool.apply_async(real_P_test, (data_c[i][:],T_max,see, data_folder, sign)))
    for i in range(data_MC.shape[0]):
        r_p_list.append(pool.apply_async(real_P_test, (data_MC[i][:],T_max,see, data_folder, sign)))
    pool.close()
    pool.join()

    for p in r_p_list:
        res = p.get()
        real_p_list.append(res)

    real_p_list_c = real_p_list[:data_c.shape[0]]
    real_p_list_MC = real_p_list[data_c.shape[0]:]
    print("真实值评估完成！")

    p_list_c = list(np.array(real_p_list_c))
    if data_c.shape[0]>1:
        data_realp_c = np.insert(data_c, 2, values=p_list_c, axis=1)
    elif data_c.shape[0]>0:
        data_realp_c = np.insert(data_c, 2, values=p_list_c)
    else:
        data_realp_c = np.array([])

    if data_realp_c.ndim == 1 and data_realp_c.size:
        data_realp_c = np.vstack((data_realp_c,-np.ones(16)))
        df = pd.DataFrame(data=data_realp_c,columns=['Cost_first','MC','Real P','[1]','[2]','[3]','[4]','[5]','[6]','[7]','[8]','[9]','[10]','[11]','[12]','[13]'])
        df.to_csv(file_)
        print("cost真实值记录完成!")
    elif data_realp_c.ndim > 1 and data_realp_c.size:
        df = pd.DataFrame(data=data_realp_c,columns=['Cost_first','MC','Real P','[1]','[2]','[3]','[4]','[5]','[6]','[7]','[8]','[9]','[10]','[11]','[12]','[13]'])
        df.to_csv(file_)
        print("cost真实值记录完成!")
    else:
        print("cost真实值为空!")

    p_list_MC = list(np.array(real_p_list_MC))
    if data_MC.shape[0]>1:     
        data_realp_MC = np.insert(data_MC, 2, values=p_list_MC, axis=1)
    elif data_MC.shape[0]>0:
        data_realp_MC = np.insert(data_MC, 2, values=p_list_MC)
    else:
        data_realp_MC = np.array([])

    if data_realp_MC.ndim == 1 and data_realp_MC.size:
        data_realp_MC = np.vstack((data_realp_MC,-np.ones(16)))
        df = pd.DataFrame(data=data_realp_MC,columns=['MC_first','MC','Real P','[1]','[2]','[3]','[4]','[5]','[6]','[7]','[8]','[9]','[10]','[11]','[12]','[13]'])
        df.to_csv(file_,mode='a')
        print("MC真实值记录完成!")
    elif data_realp_MC.ndim > 1 and data_realp_MC.size:
        df = pd.DataFrame(data=data_realp_MC,columns=['MC_first','MC','Real P','[1]','[2]','[3]','[4]','[5]','[6]','[7]','[8]','[9]','[10]','[11]','[12]','[13]'])
        df.to_csv(file_,mode='a')
        print("MC真实值记录完成!")
    else:
        print("MC真实值为空!")

    if _P >=0.99999:
        threshold_p = 0.999985
    elif _P >=0.9999:
        threshold_p = 0.99985
    elif _P >=0.999:
        threshold_p = 0.9985
    elif _P >=0.99:
        threshold_p = 0.985
    else:
        threshold_p = 0.85

    #------------ 构造最终输出解 ----------------
    if data_realp_MC.size>1:
        d_MC = data_realp_MC[:10,:]
    else:
        d_MC = data_realp_MC

    if data_realp_c.size>1:
        d_c = data_realp_c[:10,:]
    else:
        d_c = data_realp_c

    dc = []
    for i in range(d_c.shape[0]):
        if d_c[i,2]>=threshold_p:
            dc.append(d_c[i,0])

    dmc = []
    for i in range(d_MC.shape[0]):
        if d_MC[i,2]>=threshold_p:
            dmc.append(d_MC[i,0])

    with open(my_file.real_path_of(result_folder, file_3), 'a') as f:
        f.write(f"cost: {d_c.shape[0]}\n")
        f.write(f"cost num: {len(dc)}\n")
        if len(dc):
            f.write(f"cost min: {np.min(dc)}\n")
            if len(dc)>1:
                f.write(f"cost mean: {np.mean(dc)}\n")

    with open(my_file.real_path_of(result_folder, file_4), 'a') as f:
        f.write(f"MC: {d_MC.shape[0]}\n")
        f.write(f"MC num: {len(dmc)}\n")
        if len(dmc):
            f.write(f"MC min: {np.min(dmc)}\n")
            if len(dmc)>1:
                f.write(f"MC mean: {np.mean(dmc)}\n")

    data_list = []
    while len(data_list)<10 and len(d_MC)+len(d_c)>0:
        s = np.random.choice(a = [0,1], size = 1, p = [_gamma,1-_gamma])
        if s==1 and d_MC.shape[0]>0:
            _index = np.random.randint(d_MC.shape[0])
            data_list.append(d_MC[_index,:])
            d_MC = np.delete(d_MC,_index,axis=0)
        if s==0 and d_c.shape[0]>0:
            _index = np.random.randint(d_c.shape[0])
            data_list.append(d_c[_index,:])
            d_c = np.delete(d_c,_index,axis=0)

        Solu = np.array(data_list)[:,3:16]
        unique_solution_list = list(set([tuple(S) for S in Solu]))
        solution_first_index = []
        for solution in unique_solution_list:
            solution_first_index.append(list(np.where(np.all(solution==Solu,1))[0])[0]) #取出重复的第一个索引
        data_list = list(np.array(data_list)[solution_first_index,:])

    data_realp = np.array(data_list)
    if data_realp.size:
        df = pd.DataFrame(data=data_realp,columns=['Mix','MC','Real P','[1]','[2]','[3]','[4]','[5]','[6]','[7]','[8]','[9]','[10]','[11]','[12]','[13]'])
        df.to_csv(file_,mode='a') 
    else:
        print("记录失败!无可行解记录！")

    #-------------- 记录统计结果 -----------------
   
    dm = []
    for i in range(data_realp.shape[0]):
        if data_realp[i,2]>=threshold_p:
            dm.append(data_realp[i,0]) 
    
    with open(my_file.real_path_of(result_folder, file_2), 'a') as f:
        f.write(f"mix: {data_realp.shape[0]}\n")
        f.write(f"mix num: {len(dm)}\n")
        if len(dm):
            f.write(f"mix min: {np.min(dm)}\n")
            if len(dm)>1:
                f.write(f"mix mean: {np.mean(dm)}\n")

def rearrage_result_MC(data_MC, T_max, file_,file_4, data_folder, _P, seed, sign):

    see = np.random.seed(seed)
    result_folder = 'out/'

    #--------------- 评估真实值 ----------------
    r_p_list = []
    real_p_list_MC = []
    pool = Pool(processes = 10)

    for i in range(data_MC.shape[0]):
        r_p_list.append(pool.apply_async(real_P_test, (data_MC[i][:],T_max,see, data_folder, sign)))
    pool.close()
    pool.join()

    for p in r_p_list:
        res = p.get()
        real_p_list_MC.append(res)

    print("真实值评估完成！")

    p_list_MC = list(np.array(real_p_list_MC))
    if data_MC.shape[0]>1:     
        data_realp_MC = np.insert(data_MC, 2, values=p_list_MC, axis=1)
    elif data_MC.shape[0]>0:
        data_realp_MC = np.insert(data_MC, 2, values=p_list_MC)
    else:
        data_realp_MC = np.array([])

    if data_realp_MC.ndim == 1 and data_realp_MC.size:
        data_realp_MC = np.vstack((data_realp_MC,-np.ones(16)))
        df = pd.DataFrame(data=data_realp_MC,columns=['MC_first','MC','Real P','[1]','[2]','[3]','[4]','[5]','[6]','[7]','[8]','[9]','[10]','[11]','[12]','[13]'])
        df.to_csv(file_,mode='a')
        print("MC真实值记录完成!")
    elif data_realp_MC.ndim > 1 and data_realp_MC.size:
        df = pd.DataFrame(data=data_realp_MC,columns=['MC_first','MC','Real P','[1]','[2]','[3]','[4]','[5]','[6]','[7]','[8]','[9]','[10]','[11]','[12]','[13]'])
        df.to_csv(file_,mode='a')
        print("MC真实值记录完成!")
    else:
        print("MC真实值为空!记录失败!无可行解记录！")

    if _P >=0.99999:
        threshold_p = 0.999985
    elif _P >=0.9999:
        threshold_p = 0.99985
    elif _P >=0.999:
        threshold_p = 0.9985
    elif _P >=0.99:
        threshold_p = 0.985
    else:
        threshold_p = 0.85

    #------------ 构造最终输出解 ----------------
    if data_realp_MC.size>1:
        d_MC = data_realp_MC[:10,:]
    else:
        d_MC = data_realp_MC

    dmc = []
    for i in range(d_MC.shape[0]):
        if d_MC[i,2]>=threshold_p:
            dmc.append(d_MC[i,0])

    with open(my_file.real_path_of(result_folder, file_4), 'a') as f:
        f.write(f"MC: {d_MC.shape[0]}\n")
        f.write(f"MC num: {len(dmc)}\n")
        if len(dmc):
            f.write(f"MC min: {np.min(dmc)}\n")
            if len(dmc)>1:
                f.write(f"MC mean: {np.mean(dmc)}\n")

    #-------------- 记录统计结果 -----------------


