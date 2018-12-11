# -*- coding: utf-8 -*-
            
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
