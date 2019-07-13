#!/usr/bin/env python
# coding: utf-8

# In[78]:


from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import io
import time
start_time = time.clock()

# substring = {'1','2','4','5','6','9','11','12','13','14','22'}
substring = {'1'}

addr = '/home/vinay/Documents/tfs/codes/Final_code_submit/EEG_TS/session_1/'
res_addr = '/home/vinay/Documents/tfs/codes/Final_code_submit/EEG_TS/'
for ind in substring:
    address = addr + ind + '_postProcbest.mat'
    data = scipy.io.loadmat(address)

    X = data['Xproc']
    y = data['Yproc']
    X = np.swapaxes(X,1,2)



    p = np.zeros((30,X.shape[1],18))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            f, Pxx_den = signal.welch(X[i][j][:], fs=250, nperseg=125, noverlap=62, nfft=512, return_onesided=True)
            p[i][j][:] = Pxx_den[8:26]




    X_p = np.swapaxes(p,0,2)
    X_p = np.swapaxes(X_p,0,1)
    y = np.swapaxes(y,0,1)



    X_p = 10*np.log10(X_p)    #decibels


    X_p = X_p.clip(min=0)





    import pyximport
    import numpy as np
    pyximport.install(setup_args={"script_args":["--compiler=mingw32"],
                                  "include_dirs":np.get_include()},
                      reload_support=True)

    from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES
    from utils.constants import MAX_PROTOTYPES_PER_CLASS
    from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES

    from utils.utils import read_all_datasets
    from utils.utils import calculate_metrics
    from utils.utils import transform_labels
    from utils.utils import create_directory
    from utils.utils import plot_pairwise



    import random
    import utils

    from dba import calculate_dist_matrix
    from dba import dba
    from knn import get_neighbors

    def get_weights_average_selected(x_train, dist_pair_mat, distance_algorithm='dtw'):
        # get the distance function

        dist_fun = utils.constants.DISTANCE_ALGORITHMS[distance_algorithm]

        # get the distance function params
        dist_fun_params = utils.constants.DISTANCE_ALGORITHMS_PARAMS[distance_algorithm]
        # get the number of dimenions
        num_dim = x_train[0].shape[1]
        # number of time series
        n = len(x_train)
        # maximum number of K for KNN
        max_k = 5
        # maximum number of sub neighbors
        max_subk = 2
        # get the real k for knn
        k = min(max_k,n-1)
        # make sure
        subk = min(max_subk,k)
        # the weight for the center
        weight_center = 0.5
        # the total weight of the neighbors
        weight_neighbors = 0.3
        # total weight of the non neighbors
        weight_remaining = 1.0- weight_center - weight_neighbors
        # number of non neighbors
        n_others = n - 1 - subk
        # get the weight for each non neighbor
        if n_others == 0 :
            fill_value = 0.0
        else:
            fill_value = weight_remaining/n_others
        # choose a random time series
        idx_center = random.randint(0,n-1)
        # get the init dba
        init_dba = x_train[idx_center]
        # init the weight matrix or vector for univariate time series
        weights = np.full((n,num_dim),fill_value,dtype=np.float64)
        # fill the weight of the center
        weights[idx_center] = weight_center
        # find the top k nearest neighbors
        topk_idx = np.array(get_neighbors(x_train,init_dba,k,dist_fun,dist_fun_params,
                             pre_computed_matrix=dist_pair_mat,
                             index_test_instance= idx_center))
        # select a subset of the k nearest neighbors
        final_neighbors_idx = np.random.permutation(k)[:subk]
        # adjust the weight of the selected neighbors
        weights[topk_idx[final_neighbors_idx]] = weight_neighbors / subk
        # return the weights and the instance with maximum weight (to be used as
        # init for DBA )
        return weights, init_dba

    def augment_train_set(x_train, y_train, N, dba_iters=1,
                          weights_method_name = 'aa', distance_algorithm='dtw',
                          limit_N = True):
        """
        This method takes a dataset and augments it using the method in icdm2017.
        :param x_train: The original train set
        :param y_train: The original labels set
        :param N: The number of synthetic time series.
        :param dba_iters: The number of dba iterations to converge.
        :param weights_method_name: The method for assigning weights (see constants.py)
        :param distance_algorithm: The name of the distance algorithm used (see constants.py)
        """
        # get the weights function
        weights_fun = utils.constants.WEIGHTS_METHODS[weights_method_name]
        # get the distance function
        dist_fun = utils.constants.DISTANCE_ALGORITHMS[distance_algorithm]
        # get the distance function params
        dist_fun_params = utils.constants.DISTANCE_ALGORITHMS_PARAMS[distance_algorithm]
        # synthetic train set and labels
        synthetic_x_train = []
        synthetic_y_train = []

        if limit_N == True:
            # limit the nb_prototypes
            nb_prototypes_per_class = min(N, len(x_train))
        else:
            # number of added prototypes will re-balance classes
            nb_prototypes_per_class = N + (N-len(x_train))

        # get the pairwise matrix

        # loop through the number of synthtectic examples needed
        for n in range(nb_prototypes_per_class):
            # get the weights and the init for avg method
            indices = np.random.randint(x_train.shape[0], size=20)
            x_train_random = x_train[indices, :]
            y_train_random = y_train[indices, :]
            if weights_method_name == 'aa':
            # then no need for dist_matrix
                dist_pair_mat = None
            else:
                dist_pair_mat = calculate_dist_matrix(x_train_random,dist_fun,dist_fun_params)
            weights, init_avg = weights_fun(x_train_random,dist_pair_mat,
                                            distance_algorithm=distance_algorithm)

            # get the synthetic data
            synthetic_mts_x = dba(x_train_random, dba_iters, verbose=False,
                            distance_algorithm=distance_algorithm,
                            weights=weights,
                            init_avg_method = 'manual',
                            init_avg_series = init_avg)

            # add the synthetic data to the synthetic train set
            synthetic_x_train.append(synthetic_mts_x)
            new_y = 0
            sum_weights = 0
            weights_y = np.mean(weights,axis = 1)
            for s in range(y_train_random.shape[0]):
                new_y += y_train_random[s]*weights_y[s]
                sum_weights += weights_y[s]

        # update the new weighted y
            new_y = new_y/sum_weights
            synthetic_y_train.append(new_y)
            print('File',ind,'new_data_number',n)
        # return the synthetic set
        return np.array(synthetic_x_train), np.array(synthetic_y_train)




    # In[90]:


        n=100
        syn_train_set = augment_train_set(X_p, y, n ,weights_method_name='as', distance_algorithm='dtw')

        # get the synthetic train and labels
        syn_x_train, syn_y_train = syn_train_set
        # concat the synthetic with the reduced random train and labels
        aug_x_train = np.array(X_p.tolist() + syn_x_train.tolist())
        aug_y_train = np.array(y.tolist() + syn_y_train.tolist())


        # In[ ]:


        aug_data={'aug_x':aug_x_train, 'aug_y':aug_y_train}

        aug_res_address = res_addr + 'session_' + ind + '/aug_power_data_session_' +  ind +'.mat'
        io.savemat(aug_res_address,aug_data)


        original_data={'x':X_p, 'y':y}
        res_address = res_addr + 'session_' + ind + '/power_data_session_' +  ind +'.mat'

        io.savemat(res_address,original_data)
        print(time.clock()-start_time)
