# Machine Learning Project 2: Recommender Systems
The goal of this project is to apply machine learning techniques to get a very precise discrete recommender system.

## Getting started
In order to run our code, all you need to do is to type in the console at the location of run.py:
```
python run.py 
```
Note that it can be quite long to run (up to 30 mins) because it has to train multiple algorithms. We have put some print function to help to see what's going on while running the file.

 
## Dependencies
The external libraries that we use are `surprise`, `time`, `pandas`, `numpy`, `scipy`, `csv`, `tensorflow`, `keras` and `sklearn`. All of these libraries are very easy to install, and you can use the following commands if you miss some of them:
```
pip install numpy
pip install scipy
pip install pandas
pip install scikit-surprise
pip install -U scikit-learn
pip install tensorflow
```
## Required files
In order to be executed, the run.py file needs the following files in the same folder:
```
data_formatting.py
implementations.py
neural_network_regression.py
```
All those files should be already included in the code.zip file at the right location.
The data csv files (`data_train.csv` and `sampleSumbmission.csv`) must be located in the folder `csv` (so their path from `run.py` is `csv/data_train.csv` and `csv/sampleSubmission.csv`). The output `submission.csv`, and the preprocessed `data_clean.csv` files will be located in `csv` as well.

## Exact reproducibility
In order for our code to reproduce exactly our best CrowdAi submission, we set all the `random_state` parameters of both `surprise` and `sklearn` to the value `2018`. Our code should be deterministic.

## neural_network_regression.py
This file is an example of how we used neural networks for our regression. It requires tensorflow and the same dependencies as for run.py
It trains a model on a subset of our training set for regression and outputs the predictions for the remaining test set in a csv file.
