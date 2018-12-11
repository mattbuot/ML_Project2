# -*- coding: utf-8 -*-

import csv
import numpy as np
            
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
