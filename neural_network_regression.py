from submission_to_surprise import *
from surprise import SVD, Dataset, Reader
from surprise.model_selection import *
from surprise.prediction_algorithms import *
from surprise.accuracy import *
import pandas as pd
import matplotlib.pyplot as plt
from data_formatting import *
import os
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from implementations import *


PATH_CLEAN = 'csv/data_clean.csv'
PATH_SAMPLE = 'csv/sampleSubmission.csv'
PATH_SUBMISSION = 'csv/submission.csv'
data_folder = 'csv/'

PATH_ORIGINAL = 'csv/data_train.csv'
PATH_CLEAN = 'csv/data_clean.csv'
PATH_SAMPLE = 'csv/sampleSubmission.csv'
PATH_SUBMISSION = 'csv/submission.csv'

print('Reading csv file of training set')

df = pd.read_csv(PATH_CLEAN)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['User', 'Item', 'Rating']], reader)
trainset_90, testset_10 = train_test_split(data, test_size=0.1, random_state=2018)

#  Define the algorithms. We take the 3 best layer one algorithms.
#  algo_90 will train on 90% of the training set and algo_100 will train on 100% of it
algos_90 = best_algorithms(3)

print('Training the algorithms')

#  Train the algorithms
for algo in algos_90:
    learn(algo, trainset_90, verbose=True)

print('Using the algorithms on the 10% testset')

#  Make predictions for the remaining 10% of the dataset, using the algos trained on the other 90%
estimations_10 = [estimate(algo, testset_10, verbose=True) for algo in algos_90]

print('Starting to build the second layer model')

#  Put them in a DataFrame, along with the (user, item) pairs and the actual ratings
estimation_series = [pd.Series(estimation) for estimation in estimations_10]
df_test = pd.DataFrame(testset_10, columns=['User', 'Item', 'Rating'])
for i in range(0, len(algos_90)):
    column = repr(i)
    df_test[column] = estimation_series[i]

#  Gather the additional features for every (user, item) pair
users_test = [t[0] for t in testset_10]
items_test = [t[1] for t in testset_10]
df_features_test = additional_features2(users_test, items_test)

df_full = pd.concat([df_test, df_features_test], axis=1)

# Split all the data in 4 different sets between training and testing rows ; and features/labels columns
df_training_features, df_test_features, df_training_labels, df_test_labels = extract_test_train(df_full)

model = build_model()
history = model.fit(df_training_features, df_training_labels, validation_split=0.2, epochs=20, batch_size=50, verbose=1)

predictions = model.predict(df_test_features)

pd.DataFrame(predictions).to_csv('test_predictions.csv')