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
from sklearn.model_selection import GridSearchCV

from src.exception import *

def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise logger.error(f'{e}','{sys}')
    

def evaluate_models(x_train,y_train,x_test,y_test,models,params):

    """The condition that i have written here will see for parameter if the parameter is not there then it 
    will go by their base params."""

    try:
        report = {}

        for model_name, model in models.items():
            model_params = params.get(model_name, {})

            if model_params:
                gs = GridSearchCV(model, model_params, cv=3) 
                gs.fit(x_train, y_train)
                best_model = gs.best_estimator_
            else:
                best_model = model
                best_model.fit(x_train, y_train)

            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[model_name] = test_model_score


        return report
        
    except Exception as e:
        raise logger.error(f'{e}','{sys}')

