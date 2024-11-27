"""
Here we will train the model
"""

import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_msodel import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import *

from src.utils import *

@dataclass
class trainerconfig:
    trained_model_file_path= os.path.join('artifacts','model.pkl')

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config = trainerconfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Spliting trainig and test input data')

            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'RandomForest':RandomForestRegressor(),
                'Decision Tree':DecisionTreeRegressor(),
                'Gradinet Boosting':GradientBoostingRegressor(),
                'Cat Boost':CatBoostRegressor(verbose=0),
                'Ada boost':AdaBoostRegressor(),
                'K-Neighbour':KNeighborsRegressor(),
                'Linear Regression':LinearRegression(),
                'Xgb regressor':XGBRegressor()
            }

            params = {
                'RandomForest':{
                    'n_estimators':[150,175]
                }
            }

            model_report:dict=evaluate_models(x_train = X_train, y_train = y_train, x_test = X_test, y_test = y_test,models=models,params = params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            # models = list(models.values())

            best_model = models[best_model_name]  # Get the model object

            if best_model_score<0.6:
                raise logging.error('No best model found')
            else:
                save_obj(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj = best_model
                )
                predicted = best_model.predict(X_test)
                r2_Score = r2_score(y_test,predicted)
                return r2_Score    

        except Exception as e:
            return logging.error(f"Error occured: {e}",sys)