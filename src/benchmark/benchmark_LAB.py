'''
Benchmark_lab.py

生成每个因子的分布，并进行采样
    1) 模拟采样:
        采样数较小,由 sample_size 控制
    2) 真实蒙特卡洛采样:
        保留原始参数和生成器，检验时调用
    3)共有四种分布,每个因子随机挑选分布，在进行随机扰动后生成。
采样完成后保存

'''

import sys
sys.path.append(sys.path[0] + '/../')
sys.path.append("..")
import math
import random
import numpy as np
from scipy.stats import truncnorm,fatiguelife,gamma
from utils import my_file
from collections import OrderedDict

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

class Benchmark:
    
    def __init__(self,node_num,factor_num,sample_size):
        self.factor_num = factor_num # The number of items in each class
        self.node_num = node_num    # The number of classes
        self.sample_size = sample_size
        self.factor_num_sum = self.factor_num * self.node_num  # Total number of items
        self.save_folder = f'benchmark/LAB_{node_num}_{factor_num}_{sample_size}_'
        my_file.create_folder(self.save_folder)
        self.big_sample_size = 10**6

    def _save_sample(self,factor_id, samples, samplesize):
        filename_ = f'{factor_id}_sample_{samplesize}.txt'
        np.savetxt(my_file.real_path_of(self.save_folder, filename_), samples)

    def _save_attributes(self, factor_id, distribution, samples_, mu_, variance_, select = None):
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
        with open(my_file.real_path_of(self.save_folder, filename_), 'w') as f:
            for k, v in attr_dict.items():
                f.write(f'{k} {v}\n')
        print(f"Factor ID {factor_id}, Distribution {distribution}, Mean {mu_}, Variance {variance_ }, Success!")
        

    #---------------- truncated norm ----------------
    def generate_truncated_norm(self,factor_id,mean_,var_):
        mu_ = mean_*(1 + 1.2*np.random.rand()-0.6)
        variance_ = math.sqrt(var_*(1 + 1.8*np.random.rand()-0.9))
        for sam in self.sample_size:
            samples_ = Truncated_norm(sam,mu_,variance_)
            self._save_sample(factor_id, samples_, sam)

        self._save_attributes(factor_id, 'truncated_norm', samples_, mu_, variance_)

    #---------------- 双峰函数 ----------------
    def generate_bimodal(self,factor_id,mean_,var_):
        mu_ = mean_*(1 + 1.2*np.random.rand()-0.6)
        variance_ = math.sqrt(var_*(1 + 1.8*np.random.rand()-0.9))
        randomselect = np.random.randint(3)
        for sam in self.sample_size:
            samples_ = Bimodal(sam,mu_,variance_,randomselect)
            self._save_sample(factor_id, samples_, sam)
        
        self._save_attributes(factor_id, 'bimodal', samples_, mu_, variance_, select=randomselect)

    #---------------- fatiguelife ----------------
    def generate_fatiguelife(self,factor_id,mean_,var_):
        mu_ = mean_*(1 + 1.2*np.random.rand()-0.6)
        variance_ = math.sqrt(var_*(1 + 1.8*np.random.rand()-0.9))
        for sam in self.sample_size:
            samples_ = Fatiguelife(sam,mu_,variance_)
            self._save_sample(factor_id, samples_, sam)
        
        self._save_attributes(factor_id, 'fatiguelife', samples_, mu_, variance_)

    #---------------- uniform ----------------
    def generate_uniform(self,factor_id,mean_):
        mu_ = mean_*(1 + 1.2*np.random.rand()-0.6)
        variance_ = mu_**2/3
        for sam in self.sample_size:
            samples_ = Uniform(sam,mu_)
            self._save_sample(factor_id, samples_, sam)
        
        self._save_attributes(factor_id,'uniform',samples_,mu_,variance_)

    #---------------- gamma ----------------
    def generate_gamma(self,factor_id,alpha_mean_,beta_mean_):
        alpha_ = alpha_mean_*(np.random.rand() + 1)
        beta_ = beta_mean_*(np.random.rand() + 2/3)/10**3
        mu_ =  alpha_/beta_ /10**3
        variance_ = alpha_/beta_**2 /10**6
        for sam in self.sample_size:
            samples_ = Gamma_(sam,mu_,variance_)
            self._save_sample(factor_id, samples_, sam)
        
        self._save_attributes(factor_id,'gamma',samples_,mu_,variance_)

    #---------------- Randomly generate benchmark ----------------
    def generate_benchmark(self):
        
        for factor_id in range(self.factor_num_sum):          
            dist_idx = random.randint(0,4)
            if dist_idx == 0:
                self.generate_truncated_norm(factor_id,5,10)
            elif dist_idx == 1:          
                self.generate_uniform(factor_id,3)
            elif dist_idx == 2:          
                self.generate_fatiguelife(factor_id,5,10)
            elif dist_idx == 3:          
                self.generate_bimodal(factor_id,5,10)
            elif dist_idx == 4:
                self.generate_gamma(factor_id,5,6)
            else:
                raise Exception()
    #---------------- Randomly generate cost ----------------
    def generate_node_link(self):
        node_id_list = [i for i in range(self.node_num)]
        factor_id_list = [i for i in range(self.factor_num_sum)]

        node_to_factor = {}
        for node_id in node_id_list:
            res_list = []
            for factor_id in factor_id_list:
                if factor_id % self.node_num  == node_id:
                    res_list.append(factor_id)
            node_to_factor[node_id] = res_list
        for k in node_to_factor:
            print(f'{k}: {node_to_factor[k]}')

        my_file.save_pkl_in_repo(node_to_factor, self.save_folder, 'node_to_factor.pkl')
        # 方式1： 完全随机
        factor_cost_list = np.random.random(size=self.factor_num_sum) * 9 + 1
        np.savetxt(my_file.real_path_of(self.save_folder, 'cost.txt'), factor_cost_list)
        

if __name__ == '__main__':

    random.seed(827)

    # 每个benchmark在测试的时候设置两个Tmax，相同的一个置信度P0

    samplesize_list = [2,5,10,20,50,100,500,1000]
    Benchmark_1 = Benchmark(node_num=4,factor_num=4,sample_size=samplesize_list)
    # Benchmark_2 = Benchmark(node_num=3,factor_num=3,sample_size=samplesize_list)
    # Benchmark_3 = Benchmark(node_num=5,factor_num=5,sample_size=50)
    # Benchmark_4 = Benchmark(node_num=10,factor_num=10,sample_size=20)
    # Benchmark_5 = Benchmark(node_num=20,factor_num=10,sample_size=50)
    # Benchmark_6 = Benchmark(node_num=30,factor_num=10,sample_size=20)
    # Benchmark_7 = Benchmark(node_num=40,factor_num=10,sample_size=50)
    # Benchmark_8 = Benchmark(node_num=50,factor_num=10,sample_size=100)

    Benchmark_1.generate_benchmark()
    # Benchmark_2.generate_benchmark()
    # Benchmark_3.generate_benchmark()
    # Benchmark_4.generate_benchmark()
    # Benchmark_5.generate_benchmark()
    # Benchmark_6.generate_benchmark()
    # Benchmark_7.generate_benchmark()
    # Benchmark_8.generate_benchmark()

    Benchmark_1.generate_node_link()
    # Benchmark_2.generate_node_link()
    # Benchmark_3.generate_node_link()
    # Benchmark_4.generate_node_link()
    # Benchmark_5.generate_node_link()
    # Benchmark_6.generate_node_link()
    # Benchmark_7.generate_node_link()
    # Benchmark_8.generate_node_link()


