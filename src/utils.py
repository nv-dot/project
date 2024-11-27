"""
contains utility functions—small, reusable pieces of code that help with common tasks throughout
 a project. Instead of writing the same code in multiple places, you put it in utils.py so it’s 
 easy to access and keep the project organized.
"""


import os
import sys
import dill


import pandas as pd
import numpy as np
import scipy.sparse as sp
from src.logger import *

from sklearn.metrics import r2_score

from src.exception import *

def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise logger.error(f'{e}','{sys}')
    

def evaluate_models(x_train,y_train,x_test,y_test,models):

    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(x_train,y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
        
    except Exception as e:
        raise logger.error(f'{e}','{sys}')

