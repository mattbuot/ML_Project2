# -*- coding: utf-8 -*-
"""The main functions"""

import numpy as np
import scipy.sparse as sp


def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):
    """split the ratings to training data and test data.
    Args:
        min_num_ratings: 
            all users and items we keep must have at least min_num_ratings per user and per item. 
    """
    # set seed
    np.random.seed(127)
    
    # select user and item based on the condition.
    non_valid_users = np.where(num_items_per_user < min_num_ratings)[0]
    non_valid_items = np.where(num_users_per_item < min_num_ratings)[0] 
    
    num_rows, num_cols = ratings.shape
    
    valid_ratings = ratings.copy()
    valid_ratings[non_valid_items, :] = 0.0
    valid_ratings[:, non_valid_users] = 0.0
    train = sp.lil_matrix((num_rows, num_cols))
    test = sp.lil_matrix((num_rows, num_cols))
    
    print("the shape of original ratings. (# of row, # of col): {}".format(
        ratings.shape))
    
    nz_items, nz_users = ratings.nonzero()
    nz = np.array((nz_items, nz_users))
    np.random.shuffle(nz.T)
    
    nz_items_train = nz[0][:int((1-p_test)*len(nz_items))]
    nz_items_test = nz[0][int((1-p_test)*len(nz_items)):]
    nz_users_train = nz[1][:int((1-p_test)*len(nz_users))]
    nz_users_test = nz[1][int((1-p_test)*len(nz_users)):]
    
    train[nz_items_train, nz_users_train] = ratings[nz_items_train, nz_users_train]
    test[nz_items_test, nz_users_test] = ratings[nz_items_test, nz_users_test]
    train[non_valid_items, :] = 0.0
    train[:, non_valid_users] = 0.0
    
    
    print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in valid ratings:{v}".format(v=valid_ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test

def init_MF(train, num_features): 
    """init the parameter for matrix factorization."""

    num_item, num_user = train.get_shape()

    np.random.seed(988)

    Z = np.random.rand(num_features, num_user)
    W = np.random.rand(num_features, num_item)
    
    nz_items, nz_users = train.nonzero()
    global_mean = global_mean = train[nz_items, nz_users].mean()

    # start by item features.
    item_nnz = train.getnnz(axis = 1) # Donne le nombre de review pour chaque film
    item_sum = train.sum(axis = 1) # Donne la somme des notes pour chaque film
    
    for ind in range(num_item):
        if item_nnz[ind] != 0:
            W[0, ind] = item_sum[ind, 0] / item_nnz[ind] # Initialise W[0,:] avec la moyenne des notes de chaque film
        else:
            W[0, ind] = global_mean

    return Z, W

def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    
    # Utiliser un multiplication de matrice pénaliserait sur les cases vides, on est obligé d'utiliser for
    
    mse = 0
    for row, col in nz: 
        w = item_features[:, row]
        z = user_features[:, col]
        mse += (data[row, col] - (z.T @ w)) ** 2
    
    return np.sqrt(1.0 * mse / len(nz))

def matrix_factorization_SGD(train, test, num_features, lambda_user, lambda_item):
    """matrix factorization by SGD."""
    # define parameters
    gamma = 0.01
    num_epochs = 15 # number of full passes through the train set
    rmse_test_list = []
    rmse_train_list = []
    minimum_error = 5.0
    stop_criterion = 1e-4
    
    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    Z = np.copy(user_features)
    W = np.copy(item_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    #print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        for d, n in nz_train:
            w = item_features[:, d]
            z = user_features[:, n]
            err = train[d, n] - z.T.dot(w)
            
            item_features[:, d] += gamma * (err*z - lambda_item*w)
            user_features[:, n] += gamma * (err*w - lambda_user*z)

        rmse_train_list.append(compute_error(train, user_features, item_features, nz_train))
        #print("iter: {}, RMSE on training set: {}.".format(it, rmse_train_list[-1]))
        
        if len(nz_test) > 0:
            rmse_test_list.append(compute_error(test, user_features, item_features, nz_test))
            #print("RMSE on validation set: {}.".format(rmse_test_list[-1]))
        
            if (rmse_test_list[-1] < minimum_error):
                Z = user_features
                W = item_features
                minimum_error = rmse_test_list[-1]
            
            if (len(rmse_test_list) > 1) and abs(rmse_test_list[-1] - rmse_test_list[-2]) < stop_criterion:
                break
        else:
            if (rmse_train_list[-1] < minimum_error):
                Z = user_features
                W = item_features
                minimum_error = rmse_train_list[-1]
                
            if (len(rmse_train_list) > 1) and abs(rmse_train_list[-1] - rmse_train_list[-2]) < stop_criterion:
                break
        
    return Z, W, rmse_train_list, rmse_test_list

def k_fold_validation(ratings, num_items_per_user, num_users_per_item, min_num_ratings, k, test_pos):
    """split the ratings to training data and test data.
    Args:
        min_num_ratings: 
            all users and items we keep must have at least min_num_ratings per user and per item. 
    """
    # set seed
    np.random.seed(127)
    
    # select user and item based on the condition.
    non_valid_users = np.where(num_items_per_user < min_num_ratings)[0]
    non_valid_items = np.where(num_users_per_item < min_num_ratings)[0]
    
    num_rows, num_cols = ratings.shape
    
    valid_ratings = ratings.copy()
    valid_ratings[non_valid_items, :] = 0.0
    valid_ratings[:, non_valid_users] = 0.0
    
    train = sp.lil_matrix((num_rows, num_cols))
    test = sp.lil_matrix((num_rows, num_cols))
    
    print("the shape of original ratings. (# of row, # of col): {}".format(
        ratings.shape))
    
    nz_items, nz_users = ratings.nonzero()
    nz = np.array((nz_items, nz_users))
    np.random.shuffle(nz.T)
    
    n_sample = len(nz_items) # nz_items == nz_users
    n_subsample = n_sample // k
    
    indices_train1 = np.linspace(0, n_subsample * (test_pos-1) -1, n_subsample * (test_pos-1)).astype(int)
    indices_test = np.linspace(n_subsample * (test_pos-1), n_subsample*test_pos -1, n_subsample).astype(int)
    indices_train2 = np.linspace(n_subsample*test_pos, n_sample -1, n_subsample * (k - test_pos)).astype(int)
    indices_train = np.r_[indices_train1, indices_train2]
    
    nz_items_train = nz[0][indices_train]
    nz_items_test = nz[0][indices_test]
    nz_users_train = nz[1][indices_train]
    nz_users_test = nz[1][indices_test]
    
    
    train[nz_items_train, nz_users_train] = ratings[nz_items_train, nz_users_train]
    test[nz_items_test, nz_users_test] = ratings[nz_items_test, nz_users_test]
    train[non_valid_items, :] = 0.0
    train[:, non_valid_users] = 0.0
    
    
    print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in valid ratings:{v}".format(v=valid_ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test