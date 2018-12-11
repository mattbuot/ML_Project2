# -*- coding: utf-8 -*-
"""some functions for help."""

from itertools import groupby

import numpy as np
import scipy.sparse as sp
import csv


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def group_by(data, index):
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices


def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)

def get_ids_values(path):
    """Get the values and the ids from the submission"""
    
    def read_txt(path):
        """read text file from path."""
        
        with open(path, "r") as f:
            return f.read().splitlines()
            
    raw_data = read_txt(path)[1:]
    
    def deal_line(line):
        ids, values = line.split(',')
        return ids, values
        
    ids_values = [deal_line(line) for line in raw_data]
    ids = [x[0] for x in ids_values]
    values = [x[1] for x in ids_values]
    values = list(map(int, values))
        
    return ids, values
            
def create_submission(prediction):
    """Function making all the necessery step to translate the prediction matrix to lists of ids and values"""
    
    ids, values = get_ids_values("submission.csv")
    
    def replace_values_submission(ids, values, prediction):
        """Replace the 3s in the submission by the values in the prediction matrix"""
    
        for i in range(len(ids)):
            row, col = ids[i].split("_")
            row = int(row.replace("r", ""))
            col = int(col.replace("c", ""))
            
            pred = prediction[row - 1, col - 1]
            pred = int(np.round(pred))
            if pred > 5:
                pred = 5
            elif pred < 1:
                pred = 1
            else:
                pass
            values[i] = pred
            
        return values
    
    values = replace_values_submission(ids, values, prediction)
    return ids, values

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':r1,'Prediction':int(r2)})
            
def submission_to_surprise(name_train):
    """Function making all the necessery step to translate the rx_cy to actual column with user and item number so surprise      
       can work with it"""
    
    ids, values = get_ids_values(name_train)
    
    def replace_ids_submission(ids):
        """Get the Ids of the submission"""
    
        item = np.zeros((len(ids), ), dtype = 'int')
        user = np.zeros((len(ids), ), dtype = 'int')
        for i in range(len(ids)):
            row, col = ids[i].split("_")
            item[i] = int(row.replace("r", ""))
            user[i] = int(col.replace("c", ""))
            
        return item, user
    
    item, user = replace_ids_submission(ids)
    
    def create_csv_submission2(item, user, y_pred, name):
        """"""
        with open(name, 'w') as csvfile:
            fieldnames = ['Item', 'User', 'Prediction']
            writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
            writer.writeheader()
            for r1, r2, r3 in zip(item, user, y_pred):
                writer.writerow({'Item':np.squeeze(r1), 'User':np.squeeze(r2), 'Prediction':int(r3)})
            
    create_csv_submission2(item, user, values, 'surprise_train.csv')            
