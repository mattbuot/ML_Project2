# -*- coding: utf-8 -*-

import csv
import numpy as np


def read_original_csv(path):
    """Reads the original csv and returns it as a list of users, a list of items and a list of ratings"""

    with open(path, "r") as f:
        raw_data = f.read().splitlines()[1:]  # Split the lines and remove the header
        ids = [line.split(',')[0] for line in raw_data]  # Id is the first part of each line
        ratings = [int(line.split(',')[1]) for line in raw_data]  # Rating is the second part of each line

        # Each id is composed of 2 parts: user and item
        # Users have prefix 'r' for 'row' and Items have prefix 'c' for 'Column', so we have to get rid of those headers
        users = [int(line.split('_')[0][1:]) for line in ids]
        items = [int(line.split('_')[1][1:]) for line in ids]

        return users, items, ratings


def create_clean_csv(users, items, ratings, path):
    """Saves the (user,item,rating) pairs in a clean csv file"""
    with open(path, 'w') as file:
        writer = csv.DictWriter(file, delimiter=',', fieldnames=['User', 'Item', 'Rating'])
        writer.writeheader()
        for i in range(0, len(users)):
            writer.writerow({'User': users[i], 'Item': items[i], 'Rating': ratings[i]})


def clean_csv(path_original, path_clean):
    """Formats the original dataset (ruser_citem,rating) to a cleaner format with 3 columns (user,item,rating)"""

    users, items, ratings = read_original_csv(path_original)
    create_clean_csv(users, items, ratings, path_clean)


def create_csv_submission(ids, y_pred, path):
    """
    Creates an output file in csv format for submission to CrowdAI
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               path (string path of .csv output file to be created)
    """

    with open(path, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': r1, 'Prediction': int(np.round(r2))})
