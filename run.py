import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import *
from sklearn.ensemble import RandomForestClassifier
from discrete_surprise import *
from data_formatting import *
from implementations import *

PATH_ORIGINAL = 'csv/data_train.csv'
PATH_CLEAN = 'csv/data_clean.csv'
PATH_SAMPLE = 'csv/sampleSubmission.csv'
PATH_SUBMISSION = 'csv/submission.csv'

print('Cleaning the training set')

clean_csv(PATH_ORIGINAL, PATH_CLEAN)

print('Reading csv file of training set')

df = pd.read_csv(PATH_CLEAN)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['User', 'Item', 'Rating']], reader)
trainset_100 = data.build_full_trainset()
testset_100 = trainset_100.build_testset()
trainset_90, testset_10 = train_test_split(data, test_size=0.1, random_state=2018)

#  Define the algorithms. We take the 3 best layer one algorithms.
#  algo_90 will train on 90% of the training set and algo_100 will train on 100% of it
algos_90 = best_algorithms(3)
algos_100 = best_algorithms(3)

print('Training the algorithms')

#  Train the algorithms
for algo in algos_90:
    learn(algo, trainset_90, verbose=True)

for algo in algos_100:
    learn(algo, trainset_100, verbose=True)

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

#  Compute the squared error for every algorithm and for every (user, item) pair
for i in range(0, len(algos_90)):
    column = repr(i)
    df_test[column] = (df_test['Rating'] - df_test[column]) ** 2

#  We can drop the User, Item and Rating columns
df_test = df_test.drop(['User', 'Item', 'Rating'], axis=1)

#  Determine the best algorithm
df_test['Best'] = df_test.idxmin(axis=1)

#  Gather the additional features for every (user, item) pair
users_test = [t[0] for t in testset_10]
items_test = [t[1] for t in testset_10]
df_features_test = additional_features2(users_test, items_test)

#  Create the feature matrix and the classes (the classes are the algorithms' index)
X = df_features_test.values.tolist()
Y = [int(value) for value in df_test['Best']]

print('Learning the random forest')

# Train a Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=2018)
clf = clf.fit(X, Y)

print('Predicting the ratings of the sample file')

# Read the users and the items of the sample file (for which we have to predict)
users_sample, items_sample, _ = read_original_csv(PATH_SAMPLE)
df_features_sample = additional_features2(users_sample, items_sample)
classifier_results = clf.predict(df_features_sample)

#  Predict for the sample file using the results of the classifier
#  Use the algos trained on 100% of the dataset for better accuracy
predictions = predict_with_classifier(algos_100, users_sample, items_sample, classifier_results)

#  Create the submission csv file
submit(predictions, PATH_SAMPLE, PATH_SUBMISSION)

print('Done. Go to csv/submission.csv to see the results')
