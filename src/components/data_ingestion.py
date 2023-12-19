import os
import sys
from src.logging import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransfomrationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    data_repo = 'artifacts'
    train_data_path: str=os.path.join(data_repo,"train.csv")
    test_data_path: str=os.path.join(data_repo,"test.csv")
    raw_data_path: str=os.path.join(data_repo,"raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entering into data ingestion component")
        try:
            df=pd.read_csv("notebook\\data\\stud.csv")
            logging.info("Reading data set from raw csv file")

            os.makedirs(os.path.join(self.ingestion_config.data_repo),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test Split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=40)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Data Ingestion completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__== "__main__":
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array,test_array,_=data_transformation.initiate_data_transformation(train_data,test_data)    

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_array,test_array))
