'''
Benchmark_resample.py


'''

import sys
sys.path.append(sys.path[0] + '/../')
import re
import math
import random
import numpy as np
from scipy.stats import truncnorm,fatiguelife,gamma
from src.utils import my_file
from collections import OrderedDict

def data_samples(sample_size, data_folder, see, sign):
    np.random.seed(see)
    param = re.findall(r'\d+',data_folder)
    factor_num_sum = int(param[0])*int(param[1])
    prob_,ind_ = [0.9,0.09,0.009,0.001],[0,10,20,30]

    def _save_attributes(factor_id, distribution, samples_, mu_, variance_, data_folder,select = None):
        m,s = np.mean(samples_),np.std(samples_,ddof=1)
        
        attr_dict = OrderedDict()
        attr_dict['dist_type'] = distribution
        attr_dict['para_mean'] = mu_
        attr_dict['param_variance'] = variance_
        attr_dict['sample_mean'] = m
        attr_dict['sample_standard_deviation'] = s
        if select is not None:
            attr_dict['select'] = select
        filename_ = f'{factor_id}_dist.txt'
        with open(my_file.real_path_of(data_folder, filename_), 'w') as f:
            for k, v in attr_dict.items():
                f.write(f'{k} {v}\n')
        print(f"Factor ID {factor_id}, Distribution {distribution}, Mean {mu_}, Variance {variance_ }, Success!")
        _save_sample(factor_id, samples_,data_folder)

    def _save_sample(factor_id, samples,data_folder):
        filename_ = f'{factor_id}_sample.txt'
        np.savetxt(my_file.real_path_of(data_folder, filename_), samples)

    for factor_id in range(factor_num_sum):
        filename_ = f'{factor_id}_dist.txt'
        with open(my_file.real_path_of(data_folder, filename_), 'r') as f:
            data = f.readlines()
        distribution = data[0].strip().split()[1]
        mu_ = float(data[1].strip().split()[1])
        variance_ = float(data[2].strip().split()[1])

        if sign == 'APP':
            #-------------- APP 长尾数据 --------------
            if distribution == 'uniform':
                samples_ = Uniform_h(sample_size,ind_,prob_)
                _save_attributes(factor_id,'uniform',samples_,mu_,variance_,data_folder)               
            elif distribution == 'fatiguelife':
                samples_ = Fatiguelife_h(sample_size,ind_,prob_,mu_,variance_)
                _save_attributes(factor_id, 'fatiguelife', samples_, mu_, variance_,data_folder)
            elif distribution == 'bimodal':
                select = int(data[5].strip().split()[1])
                samples_ = Bimodal_h(sample_size,ind_,prob_,mu_,variance_,select)
                _save_attributes(factor_id, 'bimodal', samples_, mu_, variance_, data_folder, select=select)
            elif distribution == 'gamma':
                samples_ = Gamma_h(sample_size,mu_,variance_)
                _save_attributes(factor_id,'gamma',samples_,mu_,variance_,data_folder)
            else:
                samples_ = Truncated_norm_h(sample_size,ind_,prob_,mu_,variance_)
                _save_attributes(factor_id, 'truncated_norm', samples_, mu_, variance_,data_folder)
        else:
            #-------------- LAB 数据 --------------
            if distribution == 'uniform':
                samples_ = Uniform(sample_size,mu_)
                _save_attributes(factor_id,'uniform',samples_,mu_,variance_,data_folder)               
            elif distribution == 'fatiguelife':
                samples_ = Fatiguelife(sample_size,mu_,variance_)
                _save_attributes(factor_id, 'fatiguelife', samples_, mu_, variance_,data_folder)
            elif distribution == 'bimodal':
                select = int(data[5].strip().split()[1])
                samples_ = Bimodal(sample_size,mu_,variance_,select)
                _save_attributes(factor_id, 'bimodal', samples_, mu_, variance_, data_folder, select=select)
            elif distribution == 'gamma':
                samples_ = Gamma_(sample_size,mu_,variance_)
                _save_attributes(factor_id,'gamma',samples_,mu_,variance_,data_folder)
            else:
                samples_ = Truncated_norm(sample_size,mu_,variance_)
                _save_attributes(factor_id, 'truncated_norm', samples_, mu_, variance_,data_folder)


def Uniform(sample_size,mu_):

    low_,high_ = 0, 2*mu_
    return np.random.uniform(low_, high_, sample_size)

def Fatiguelife(sample_size,mu_,variance_):
    
    scale_ = (4 * mu_-math.sqrt(mu_**2 + 3*variance_)) / 3
    shape_ = math.sqrt(2 * (math.sqrt(mu_**2 + 3*variance_)-mu_) / (4 * mu_-math.sqrt(mu_**2+3*variance_)))
    return fatiguelife.rvs(shape_, loc=0, scale=scale_, size=sample_size, random_state=None)

def Truncated_norm(sample_size,mu_,variance_):

    sigma_ = math.sqrt(variance_)
    clip_a, clip_b = 0, np.inf
    a,b = (clip_a - mu_) / sigma_, (clip_b - mu_) / sigma_
    return truncnorm.rvs(a, b, loc=mu_, scale=sigma_, size=sample_size)

def Bimodal(sample_size,mu_,variance_,select):

    s_ = math.sqrt(variance_)
    if select == 0:
    # 构造合适的双峰函数
        mu_ =[mu_ + s_/2, mu_ - 3*s_/2]
        sigma_= [s_/2, s_/2]
        choice_1 = list(np.random.choice(a = [0,1], size = sample_size, p = [0.6,0.4]))
        sample_size_1 = choice_1.count(0)
        sample_size_2 = choice_1.count(1)
    elif select == 1:
        mu_ =[mu_ + 3*s_/2, mu_ - s_/2]
        sigma_= [s_/2, s_/2]
        choice_1 = list(np.random.choice(a = [0,1], size = sample_size, p = [0.4,0.6]))
        sample_size_1 = choice_1.count(0)
        sample_size_2 = choice_1.count(1)
    else:
        mu_ =[mu_ + s_, mu_ - s_]
        sigma_= [s_/4, s_/4]
        choice_1 = list(np.random.choice(a = [0,1], size = sample_size, p = [0.5,0.5]))
        sample_size_1 = choice_1.count(0)
        sample_size_2 = choice_1.count(1)

    clip_a, clip_b = 0, np.inf
    samples_1 = truncnorm.rvs((clip_a - mu_[0]) / sigma_[0], (clip_b - mu_[0]) / sigma_[0], loc=mu_[0], scale=sigma_[0], size=sample_size_1)
    samples_2 = truncnorm.rvs((clip_a - mu_[1]) / sigma_[1], (clip_b - mu_[1]) / sigma_[1], loc=mu_[1], scale=sigma_[1], size=sample_size_2)
    return np.concatenate([samples_1,samples_2])

def Gamma_(sample_size, mu_,variance_):
    alpha_ = mu_**2/variance_
    beta_ = mu_/(variance_*10**3)

    return gamma.rvs(alpha_, loc=0, scale=1/beta_, size=sample_size, random_state=None)/1000

def Uniform_h(sample_size,ind_,prob):

    low_,high_ = 0,10
    samples_ = np.random.uniform(low_, high_, sample_size)    
    choice_ = np.random.choice(a = ind_, size = len(samples_), p = prob)
    samples_ = np.sum([samples_,choice_],axis=0).tolist()
    return samples_

def Fatiguelife_h(sample_size,ind_,prob,mu_,variance_):

    scale_ = (4 * mu_-math.sqrt(mu_**2 + 3*variance_)) / 3
    shape_ = math.sqrt(2 * (math.sqrt(mu_**2 + 3*variance_)-mu_) / (4 * mu_-math.sqrt(mu_**2+3*variance_)))
    samples_ = fatiguelife.rvs(shape_, loc=0, scale=scale_, size=sample_size*2, random_state=None)
    samples_ = list(filter(lambda i: i<=10,samples_))
    choice_ = np.random.choice(a = ind_, size = len(samples_), p = prob)
    samples_ = np.sum([samples_,choice_],axis=0).tolist()
    return samples_[:sample_size]

def Truncated_norm_h(sample_size,ind_,prob,mu_,variance_):

    sigma_ = math.sqrt(variance_)
    clip_a, clip_b = 0, 10
    a,b = (clip_a - mu_) / sigma_, (clip_b - mu_) / sigma_
    samples_  = truncnorm.rvs(a, b, loc=mu_, scale=sigma_, size=sample_size)
    choice_ = np.random.choice(a = ind_, size = len(samples_), p = prob)
    samples_ = np.sum([samples_,choice_],axis=0).tolist()
    return samples_

def Bimodal_h(sample_size,ind_,prob,mu_,variance_,select):

    s_ = math.sqrt(variance_)
    if select == 0:
    # 构造合适的双峰函数
        mu_ =[mu_ + s_/2, mu_ - 3*s_/2]
        sigma_= [s_/2, s_/2]
        choice_1 = list(np.random.choice(a = [0,1], size = sample_size, p = [0.6,0.4]))
        sample_size_1 = choice_1.count(0)
        sample_size_2 = choice_1.count(1)
    elif select == 1:
        mu_ =[mu_ + 3*s_/2, mu_ - s_/2]
        sigma_= [s_/2, s_/2]
        choice_1 = list(np.random.choice(a = [0,1], size = sample_size, p = [0.4,0.6]))
        sample_size_1 = choice_1.count(0)
        sample_size_2 = choice_1.count(1)
    else:
        mu_ =[mu_ + s_, mu_ - s_]
        sigma_= [s_/4, s_/4]
        choice_1 = list(np.random.choice(a = [0,1], size = sample_size, p = [0.5,0.5]))
        sample_size_1 = choice_1.count(0)
        sample_size_2 = choice_1.count(1)

    clip_a, clip_b = 0, 10
    samples_1 = truncnorm.rvs((clip_a - mu_[0]) / sigma_[0], (clip_b - mu_[0]) / sigma_[0], loc=mu_[0], scale=sigma_[0], size=sample_size_1)
    samples_2 = truncnorm.rvs((clip_a - mu_[1]) / sigma_[1], (clip_b - mu_[1]) / sigma_[1], loc=mu_[1], scale=sigma_[1], size=sample_size_2)
    samples_ = np.concatenate([samples_1,samples_2])
    choice_2 = np.random.choice(a = ind_, size = len(samples_), p = prob)
    samples_ = np.sum([samples_,choice_2],axis=0).tolist()   # 构筑长尾分布
    return samples_

def Gamma_h(sample_size, mu_,variance_):
    alpha_ = mu_**2/variance_
    beta_ = mu_/(variance_*10**3)

    return gamma.rvs(alpha_, loc=0, scale=1/beta_, size=sample_size, random_state=None)/1000
        

if __name__ == '__main__':

    see = random.seed(327)
    sign_list = ['LAB','APP']
    for s in sign_list:
        if s == 'LAB':
            Instance_folder = [\
                'benchmark/LAB_3_5_30_',
                'benchmark/LAB_4_5_30_',
                'benchmark/LAB_5_5_30_',
                'benchmark/LAB_10_10_500_',
                'benchmark/LAB_20_10_500_',
                'benchmark/LAB_30_10_500_',
                'benchmark/LAB_40_10_500_',
                'benchmark/LAB_50_10_500_']  
        else:
            Instance_folder = [\
                'benchmark/APP_3_5_30_',
                'benchmark/APP_4_5_30_',
                'benchmark/APP_5_5_30_',
                'benchmark/APP_10_10_500_',
                'benchmark/APP_20_10_500_',
                'benchmark/APP_30_10_500_',
                'benchmark/APP_40_10_500_',
                'benchmark/APP_50_10_500_']
           
        for i in range(len(Instance_folder)):
            param = re.findall(r'\d+',Instance_folder[i])
            if i==0:
                sample_size = 30
            if i==1:
                sample_size = 30
            if i==2:
                sample_size = 30
            if i==3:
                sample_size = 500
            if i==4:
                sample_size = 500
            if i==5:
                sample_size = 500
            if i==6:
                sample_size = 500
            if i==7:
                sample_size = 500
            data_samples(sample_size, Instance_folder[i], see, s)

