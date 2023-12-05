'''
定义因子的类
'''

import numpy as np

max_K = 10

class Factor:

    def __init__(self, factor_id, cost, value, sample_filepath, big_sample_filepath=None):
        self.id = factor_id  # 独特id
        self.cost = cost
        self.value = value
        self.sample_filepath = sample_filepath  # sample对应的路径
        self.samples = np.loadtxt(sample_filepath)  # 采样数据，ndarray类  
        self.big_sample_filepath = big_sample_filepath  # 大规模采样对应的路径

        self._sample_min = None
        self._sample_max = None

        self._sample_mean = None
        self._sample_var = None
        self._sample_std = None
        self._big_samples = None #  np.loadtxt(big_sample_filepath)

        self._moments_min = None
        self._moments_origin = None
        # self._rnd_indices = np.arange(len(self.samples))

        self.weight = None
        self.utility = None
        self.increment = None

    def resample_once(self):
        # 采样一次，返回采样结果 (长度为1的ndarray)
        return self.resample_multi(1)

    def resample_multi(self, sample_number):
        # 采样 sample_number 次, 返回采样结果 (长度为 sample_number 的ndarray)
        _idx = np.random.randint(0, len(self.samples) - 1, sample_number)

        return self.samples[_idx]

    def resample_big_multi(self, sample_number):
        # 采样 sample_number 次, 返回采样结果 (长度为 sample_number 的ndarray)
        _idx = np.random.randint(0, len(self._big_samples) - 1, sample_number)

        return self._big_samples[_idx]

    # 创建只读属性
    @property
    def sample_min(self):
        # 采样最小值
        if self._sample_min is None:
            self._sample_min = np.min(self.samples)

        return self._sample_min

    @property
    def sample_max(self):
        # 采样最大值
        if self._sample_max is None:
            self._sample_max = np.max(self.samples)

        return self._sample_max

    @property
    def sample_mean(self):
        # 采样均值
        if self._sample_mean is None:
            self._sample_mean = np.mean(self.samples)

        return self._sample_mean

    @property
    def sample_var(self):
        # 采样方差
        if self._sample_var is None:
            self._sample_var = np.var(self.samples)

        return self._sample_var

    @property
    def sample_std(self):
        # 采样标准差
        if self._sample_std is None:
            self._sample_std = np.std(self.samples,ddof=1)

        return self._sample_std

    @property
    def big_samples(self):
        # 大规模采样，华为给真实数据后这项无用
        if self._big_samples is None:
            self._big_samples = np.loadtxt(self.big_sample_filepath)

        return self._big_samples

    def n_moment_min(self, K):
        # 计算减去最小值的1~K阶矩
        sample_size = len(self.samples)
        _aux_matrix = np.ones((2, sample_size))
        moments_list = np.ones(K + 1)

        for k in range(1, K + 1): # 1 to K
            _cur_idx = k % 2
            _prev_idx = 1 - _cur_idx
            _aux_matrix[_cur_idx] = (self.samples - self.sample_min) * _aux_matrix[_prev_idx]

            moments_list[k] = np.mean(_aux_matrix[_cur_idx])

        return moments_list

    def n_moment_of_origin(self, K):
        # 计算1~K阶原点矩
        sample_size = len(self.samples)
        _aux_matrix = np.ones((2, sample_size))
        moments_list = np.ones(K + 1)

        for k in range(1, K + 1): # 1 to K
            _cur_idx = k % 2
            _prev_idx = 1 - _cur_idx
            _aux_matrix[_cur_idx] = self.samples * _aux_matrix[_prev_idx]

            moments_list[k] = np.mean(_aux_matrix[_cur_idx])

        return moments_list

    @property
    def moments_min(self):
        # 减去最小值的1~K阶矩
        if self._moments_min is None:
            self._moments_min = self.n_moment_min(max_K)

        return self._moments_min

    @property
    def moments_origin(self):
        # 1~K阶原点矩
        if self._moments_origin is None:
            self._moments_origin = self.n_moment_of_origin(max_K)

        return self._moments_origin

    def set_weight(self,weight):
        self.weight = weight

    def set_utility(self,utility):
        self.utility = utility

    def set_increment(self,increment):
        self.increment = increment

    def get_index_sample(self,_l):
        # 返回数据中第L大的数据，最大为0，最小为L-1
        return np.sort(self.samples)[::-1][_l]