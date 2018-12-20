from surprise import *
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import *
from surprise.prediction_algorithms import *
from discrete_surprise import *
import time
import pandas as pd
import numpy as np


def best_algorithms(limit=None):
    """Returns the best algorithms.
    If you want to restrict the number of algorithms, you can specify the limit parameter"""

    return [SVDDiscrete(n_epochs=60, n_factors=40, biased=True, random_state=2018, reg_all=0.05, lr_all=0.002),
            KNNBaselineDiscrete(k=150, min_k=2, sim_options={'user_based': True}, verbose=False),
            SlopeOneDiscrete(),
            BaselineOnlyDiscrete(bsl_options={'reg_i': 10, 'reg_u': 15, 'n_epochs': 10}, verbose=False),
            KNNWithMeansDiscrete(k=60, min_k=2, sim_options={'user_based': True}, verbose=False),
            KNNWithZScoreDiscrete(k=70, min_k=2, sim_options={'user_based': True}, verbose=False),
            CoClusteringDiscrete(n_cltr_u=3, n_cltr_i=3, n_epochs=20, random_state=2018),
            KNNBasicDiscrete(k=40, min_k=2, sim_options={'user_based': True}, verbose=False),
            NMFDiscrete(n_factors=15, n_epochs=50, biased=False, reg_pu=0.06, reg_qi=0.06, reg_bu=0.02, reg_bi=0.02,
                        lr_bu=0.005, lr_bi=0.005, init_low=0, init_high=1, random_state=2018)][:limit]


def learn(algo, trainset, testset=None, verbose=False):
    """Learns the specified algorithm on the specified training set. If a test set is given, it will also compute
    the RMSE of the algorithm on the test set. If verbose is set to True, it will print the learning time"""

    start = time.process_time()
    algo.fit(trainset)
    training_time = time.process_time() - start
    if testset != None:
        rmse = accuracy.rmse(algo.test(testset), verbose=False)
        testing_time = time.process_time() - start - training_time
        if verbose:
            print('Algo learnt in ' + repr(training_time) + ' seconds, with RMSE of '
                  + repr(rmse) + ' computed in ' + repr(testing_time) + ' seconds')
    else:
        if verbose:
            print('Algo learnt in ' + repr(training_time) + ' seconds')


def estimate(algo, testset, verbose=False):
    """Computes the estimations of a specified algorithm on a specified test set. If verbose is set to True, it will
    print the estimation time"""

    start = time.process_time()
    estimations = [prediction.est for prediction in algo.test(testset)]
    estimation_time = time.process_time() - start
    if verbose:
        print('Estimations computed in ' + repr(estimation_time) + ' seconds')
    return estimations


def additional_features1(df, users, items):
    """Compute the set of additional features for some specified (user, item) pairs,
    with the help of the training DataFrame. For each (user, item) pair, these features are:
        -user's standard deviation
        -user's number of ratings
        -item's standard deviation
        -item's number of ratings"""

    user_std = df.groupby('User')['Rating'].std()
    user_count = df.groupby('User')['Rating'].count()
    item_std = df.groupby('Item')['Rating'].std()
    item_count = df.groupby('Item')['Rating'].count()

    return pd.DataFrame({'u_std': user_std[users].values, 'u_count': user_count[users].values,
                        'i_std': item_std[items].values, 'i_count': item_count[items].values})


def predict_with_classifier(algos, users, items, classifier_results):
    """Makes the predictions for some specified (user, item) pairs, with some specified algorithms.
    classifier_results determines which algorithm to use for each (user, item) pair,
    based on the results of the second layer of machine learning"""

    predictions = np.ones(len(items))
    for i in range(len(items)):
        algo_to_use = algos[classifier_results[i]]
        predictions[i] = algo_to_use.predict(users[i], items[i]).est
    return predictions


#  The functions below are unused in our final solution


def additional_features2(users, items):
    """Compute the set of additional features for some specified (user, item) pairs,
        with the help of the training DataFrame. For each (user, item) pair, these features are:
            -user
            -item
        (This is a much simpler version of additional_features1)"""

    return pd.DataFrame({'User': users, 'Item': items})


def additional_features3(users):
    """Compute the set of additional features for some specified (user, item) pairs,
        with the help of the training DataFrame. For each (user, item) pair, these features are:
            -user
        (This is an even simpler version of additional_features2)"""

    return pd.DataFrame({'User': users})
