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

@dataclass
class dataingestionconfig():
    test_data_path: str=os.path.join('artifacts/test_data.csv')
    train_data_path: str=os.path.join('artifacts/train_data.csv')
    original_data_path: str=os.path.join('artifacts/data.csv')

class dataingestion():
    def __init__(self):
        self.ingestion_config = dataingestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method')
        try:
            df = pd.read_csv('notebook/data/stud.csv')
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
    obj.initiate_data_ingestion()