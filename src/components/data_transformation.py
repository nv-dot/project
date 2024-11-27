"""
Here we will transform the data
"""

import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import *
from src.logger import *
from src.utils import save_obj


@dataclass
class datatransformationconfig:
    preprocess_obj_file = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = datatransformationconfig()

    def get_data_transformation_obj(self):

        """
        This function will transform the columns so that we can use in for model building
        """

        try :
            numerical_columns= ['math_score', 'reading_score', 'writing_score']
            cat_columns = ['gender', 'race_ethnicity', 
                           'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False)),
                ]
            )

            logging.info('Numerical column Transformation is complete')

 
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info('Categorical column Transformation is complete')

            logging.info(f'Num columns are {numerical_columns}')
            logging.info(f'Cat columns are {cat_columns}')

            preprocesser = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('Cat_pipeline',cat_pipeline,cat_columns)
                ]
            )
                              
            return preprocesser

        except Exception as e:
            raise handle_exception(e,sys)
        
    def initiate_datatransformation(self,train_path,test_path):
        try:
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            unwanted_columns = ['Unnamed: 0.1', 'Unnamed: 0']
            train = train.drop(columns=unwanted_columns,axis=1)
            test = test.drop(columns=unwanted_columns,axis=1)

            logging.info('Train and test data has been read')

            logging.info('Obtaining preprocessing objects')

            preprocessor_obj = self.get_data_transformation_obj()
            target_column = 'total_score'

            numerical_col = ['math_score', 'reading_score', 'writing_score']
            cat_columns = ['gender', 'race_ethnicity', 
                           'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            input_feature_train_df = train.drop(columns=[target_column],axis=1)
            target_feature_train_df = train[target_column]

            input_feature_test_df = test.drop(columns=[target_column],axis=1)
            target_feature_test_df = test[target_column]

            logging.info('Applying our preprocess obj in train and test data')


            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info('Saved preprocessor objects')

            save_obj(
                file_path = self.data_transformation_config.preprocess_obj_file,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_obj_file  
            )


        except Exception as e:
            raise handle_exception(e,sys)

        
