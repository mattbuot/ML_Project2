
# coding: utf-8

# In[1]:


from helpers import *
from plots import *
from main_functions import *
import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
from surprise import *
from surprise.model_selection import cross_validate


# In[2]:


reader = Reader(line_format='user item rating', sep=',', skip_lines = 1)
data = Dataset.load_from_file("surprise_train.csv", reader=reader)


# In[3]:


trainset = data.build_full_trainset()
#Paramètres optimaux trouvés avec une grid search
algo = SVD(100, 20, True, 0, 0.1, 0.005, 0.02, None, None, None, None, 0.01, 0.1, 0.1, 0.01, None, True)
algo.fit(trainset)


# In[4]:


ids, _ = get_ids_values('csv/sampleSubmission.csv')
    
def replace_ids_submission(ids): #La fonction est déjà définie dans submission_to_surprise
    """Get the Ids of the submission"""
    
    item = np.zeros((len(ids), ), dtype = 'int')
    user = np.zeros((len(ids), ), dtype = 'int')
    for i in range(len(ids)):
        row, col = ids[i].split("_")
        item[i] = int(row.replace("r", ""))
        user[i] = int(col.replace("c", ""))
            
    return item, user
item, user = replace_ids_submission(ids)


# In[5]:


prediction = np.zeros(len(item), )
for i in range(len(prediction)):
    prediction[i] = algo.predict(str(user[i]), str(item[i]), None, True, False).est


# In[6]:


create_csv_submission(ids, prediction, 'test2.csv')

