"""
Here in this file we will read the data from different database or from different location
"""

import os
import sys
from src.exception import *
from src.logger import *

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import trainerconfig, Modeltrainer

@dataclass
class dataingestionconfig():
    test_data_path: str=os.path.join('artifacts/test_data.csv')
    train_data_path: str=os.path.join('artifacts/train_data.csv')
    original_data_path: str=os.path.join('artifacts/data_new_.csv')

class dataingestion():
    def __init__(self):
        self.ingestion_config = dataingestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method')
        try:
            df = pd.read_csv('notebook/data_new.csv')
            logger.info('Dataset has been read and converted into a df')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.original_data_path, index=True, header= True)

            logger.info('Train test spilt is initiated')

            train_set, test_set= train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=True, header= True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=True, header= True)

            logger.info('Ingestion of the data is completed.')

            return(self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)


        except Exception as e:
            raise handle_exception(e,sys)
        
if __name__ == '__main__':
    obj = dataingestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_datatransformation(train_data,test_data)

    model_trainer = Modeltrainer()
    print(model_trainer.initiate_model_training(train_arr,test_arr))
