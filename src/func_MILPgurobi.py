'''
一次性声明多个变量
一次性添加所有约束
'''
import gurobipy as gp
import numpy as np
import time
import re
import sys
sys.path.append(sys.path[0] + '/../')
sys.path.append("..")

from utils import my_file
from utils.factor import Factor
# from func_realP_statistical import real_P_test
from func_solution_factor import solution_cost,Statistic_factor
from gurobipy import GRB

import func_timeout
from func_timeout import func_set_timeout
 
@func_set_timeout(1000) # 设置函数最大执行时间
 

def Statistic_factor(factor_num, item_num, samplesize, data_folder):
    '''
    Statistic_factor: 给出所有item的必要统计信息及重排序
    data_folder: benchmark数据存放文件夹

    输出: Factor类的所有Item
    '''
    item_num_sum = item_num * factor_num
    item_id_list = [i for i in range(item_num_sum)]

    cost_list = np.loadtxt(my_file.real_path_of(data_folder, 'cost.txt'))
    max_cost = np.max(cost_list)

    item2obj = [] # 映射：因子id->具体因子

    for item_id in item_id_list:
        sample_filename_ = f'{item_id}_sample.txt'
        big_sample_filename_ = f'{item_id}_big_sample.txt'

        sample_path_ = my_file.real_path_of(data_folder, sample_filename_)
        big_sample_path_ = my_file.real_path_of(data_folder, big_sample_filename_)

        cost_ = cost_list[item_id]
        value_ = max_cost - cost_list[item_id]

        cur_factor = Factor(item_id, cost_, value_, sample_path_, big_sample_path_)
        item2obj.append(cur_factor)

    print(f'\n====================== Read Complete! {len(item2obj)} Items ===================')

    return item2obj

def Reorganized_FTO(nn,fto):
    # 把FactortoObject由[0,1,2,...,14]转化为[0,3,6,9,...,14]
    node_id_list = [i for i in range(nn)]
    Orga_FTO = []
    
    for node_id in node_id_list:
        res_list = []
        for factor in fto:
            if factor.id % nn  == node_id:
                res_list.append(factor)
        Orga_FTO = Orga_FTO + res_list
    
    return Orga_FTO

def Constrct_Randomxi_Cost(Orga_FTO):
    xi = []
    _cost = []

    for fto in Orga_FTO:
        xi.append(-fto.samples)     # 取负
        _cost.append(fto.cost)

    return np.array(xi),np.array(_cost)

def Enumerate_Objectx(factor_num,item_num):
    """
    获得x的全部0-1组合共有2**MN个可能
    子函数：获取十进制数字的固定长度二进制表示向量

    :param decimal: 十进制数字
    :param length: 二进制向量的长度
    :return: x的全部0-1组合
    """
    Enu_x = []
    length_x = factor_num*item_num
    for decimal in range(2**length_x):
        binary = bin(decimal)[2:]  # 将十进制数字转换为二进制字符串，去掉前缀“0b”
        binary_vector = [int(bit) for bit in binary.zfill(length_x)]  # 将二进制字符串转换为长度为length的二进制向量
        Enu_x.append(binary_vector)

    return np.array(Enu_x)

def concrate(x,y):
    result = []
    for x_li in x:
        for y_li in y:
            new_list = x_li + y_li
            result.append(new_list)
    return result

def Enumerate_Objectxx(factor_num,item_num):
    """
    获得x的全部0-1组合共有2**MN个可能
    子函数：获取十进制数字的固定长度二进制表示向量

    :param decimal: 十进制数字
    :param length: 二进制向量的长度
    :return: x的全部0-1组合
    """
    
    x = []
    for i in range(item_num):
        x_list = [0 for _ in range(item_num)]
        x_list[i] = x_list[i] + 1
        x.append(x_list)

    y = x.copy()
    result = []
    for i in range(factor_num-1):
        result = concrate(x,y)
        x = result.copy()

    return np.array(result)


def Calculate_UpperBoundEta(factor_num, item_num, Tau_e, Tmax, T_sample, X_enumerate):
    '''
    param: factor_num   # 环节数
    param: item_num   # 因子数
    param: Tau_e # 人为定义的非负值
    '''
    # 输入第l个样本

    # X_enumerate 是np.array格式
    sign = 0
    count = 0
    array_xi = T_sample # 时延样本变量， (M*N) 一维
    b = -Tmax   # 取负
    M = [m for m in range(factor_num*item_num)]

    for x_e in X_enumerate:
        feasible_detect = 1
        for u in range(factor_num):
            if sum(x_e[i] for i in range(u*item_num, (u+1)*item_num)) != 1:
                feasible_detect = 0
                break
            else:
                continue
        if feasible_detect == 1:
            count += 1
            if np.dot(x_e,array_xi) - b > 0:
                sign = 1
                break

    if sign:
        # 创建模型对象
        model = gp.Model('LP for UpperBoundEta')
        model.setParam('OutputFlag', 0) # 关闭输出显示

        # 设置变量
        x = model.addVars(factor_num*item_num, lb=0, ub=1, vtype=gp.GRB.BINARY, name="X")    # 0-1整型变量
        model.update()

        # 设置目标函数
        model.setObjective(gp.quicksum(x[m] * array_xi[m] for m in M) - b, sense=gp.GRB.MINIMIZE)

        # 添加约束条件
        model.addConstr(b - gp.quicksum(x[m] * array_xi[m] for m in M) <= 0 - Tau_e, name="c0")
        for u in range(factor_num):
            model.addConstr(gp.quicksum(x[i] for i in range(u*item_num, (u+1)*item_num)) == 1)

        # 运行求解器并获取结果
        model.optimize()

        # print("x =", x.X)
        # print("y =", y.X)
        # print("Objective value =", m.ObjVal)
        UpperBoundEta = model.ObjVal
    else:
        UpperBoundEta = 0

    return count,UpperBoundEta


def MILP_CCMCKP(factor_num, item_num, sample_size, Ep, Delta_e, Radius_, Tmax, T_sample_total, Cost, UpperBound_list):
    '''
    param: factor_num   # 环节数
    param: item_num   # 因子数
    param: sample_size # 样本量
    param: Epilon # 1-问题置信度
    param: Delta_e # 无穷小的正数
    param: Radius_ # 沃瑟斯坦距离
    param: Tmax    # 时延上界
    param: T_sample_total   # 时延样本变量， (M*N)*L二维
    '''

    b = -Tmax   # 取负
    b_e = b - Delta_e
    _XI = T_sample_total
    _cost = Cost    # 因子成本，M*N
    M = [m for m in range(factor_num*item_num)]
    L = [l for l in range(sample_size)]
    

    # 创建模型对象
    model = gp.Model('MILP for CC-MCKP')

    # 设置变量
    x = model.addVars(factor_num*item_num, lb=0, ub=1, vtype=gp.GRB.BINARY, name="X")    # 0-1整型变量
    _Lambda = model.addVar(vtype=gp.GRB.CONTINUOUS, name="LAMBDA")  # 默认下界为0，上界为无穷大的连续变量
    s = model.addVars(sample_size, vtype=gp.GRB.CONTINUOUS, name="s")
    Eta = model.addVars(sample_size, vtype=gp.GRB.CONTINUOUS, name="Eta")
    Phi = model.addVars(factor_num*item_num, sample_size, vtype=gp.GRB.CONTINUOUS, name="Phi")

    model.update()

    # 设置目标函数
    model.setObjective(gp.quicksum(x[m] * _cost[m] for m in M), sense=gp.GRB.MINIMIZE)

    # 添加约束条件
    model.addConstr(_Lambda*Radius_ + gp.quicksum(s)/sample_size <= Ep, name="c1")
    model.addConstrs(1 + Eta[l]*b_e -  gp.quicksum(_XI[m,l]* Phi[m,l] for m in M) <= s[l] for l in L)
    model.addConstrs(Phi[m,l] == x[m] * Eta[l] for m in M for l in L)
    model.addConstrs(Phi[m,l] <= _Lambda for m in M for l in L)
    model.addConstrs(Phi[m,l] <= Eta[l] for m in M for l in L)
    model.addConstrs(Phi[m,l] <= UpperBound_list[l] * x[m] for m in M for l in L)
    model.addConstrs(Phi[m,l] >= Eta[l] + UpperBound_list[l] * x[m] - UpperBound_list[l] for m in M for l in L)

    for u in range(factor_num):
        model.addConstr(gp.quicksum(x[i] for i in range(u*item_num, (u+1)*item_num)) == 1)

    # 运行求解器并获取结果
    model.optimize()

    vars = model.getVars()
    # for v in vars[:len(M)]:
    #     print(v.VarName, v.X)
    # model.computeIIS()
    x_c=np.zeros((item_num,factor_num))
    # 检查是否找到了最优解
    if model.status == GRB.OPTIMAL:
        for j in range(factor_num):
            for i in range(item_num):
                x_c[i,j] = x[j * item_num + i].x

        solution = []
        for i in range(factor_num):
            solution.append(np.where(x_c[:,i]==1)[0][0] * factor_num + i) 
        
        #输出结果
        return model.ObjVal, solution
    else:
        # 处理未找到最优解的情况
        print("Gurobi未找到最优解，状态码：", model.status)
        return None,[]


if __name__=='__main__':
    '''
    MILP方法
    '''
    folder_name = 'benchmark/Instance_lab_4_5_30_'
    param = re.findall(r'\d+',folder_name)
    Tmax = 35
    Epsilon = 1-0.9  # 1-问题置信度
    tau_ = 0.01   # 参数，小的非负数
    delta_ = 0.001  # 参数，无穷小的正数
    radius_ = 0.03    # 沃瑟斯坦距离参数

    factor_num = int(param[0])
    item_num = int(param[1])
    #samplesize_list = [2,5,10,20,50,100,500,1000]
    sample_size = int(param[2])

    ITEMtoObject = Statistic_factor(factor_num,item_num, sample_size, folder_name)
    Organized_FTO = Reorganized_FTO(factor_num,ITEMtoObject)
    XI,Cost = Constrct_Randomxi_Cost(Organized_FTO)

    try:
        # 在这里编写您的代码
        X_enumerate = Enumerate_Objectxx(factor_num,item_num)
        se = 233
        start = time.time()
        upper_eta = []
        cc = []
        for l in range(sample_size):
            c,u = Calculate_UpperBoundEta(factor_num, item_num, tau_, Tmax, XI[:,l], X_enumerate)
            upper_eta.append(u)
            cc.append(c)

        print("Upper Bounds for dual variable Eta are calculated!")
        print("********* Time: ", time.time()-start)
        print("==================================================")

        obj_val, x_ = MILP_CCMCKP(factor_num, item_num, sample_size, Epsilon, delta_, radius_, Tmax, XI, Cost, upper_eta)
        
        print("MILP for CC-MCKP is Done!")
        print("********* Time: ", time.time()-start)
        print(obj_val,x_)

        # print(real_P_test(x_, Tmax, se, folder_name, 'lab'))

        # print(upper_eta)

    except func_timeout.exceptions.FunctionTimedOut:
        print("超时了,自动退出")
    
