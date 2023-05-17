import os
import sys
from src.exception import Custom_Exception
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 
from src.components.data_transformation import DataTransformationConfig,DataTranformation
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion has started.")
        try:
            df=pd.read_csv(r'C:\Users\Admin\ML_Projects\Predict_HR_Employee_Joining_Company\Predict-HR-Employee-Joining-Company-Using-ML\src\notebook\data.csv')
            logging.info('Reading the data has started')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Train Test Split has Started.')
            train_set,test_set = train_test_split(df,
                                                    test_size=0.25,
                                                    random_state=42,
                                                    )
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Train test Split has been done successfully.')  
            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)    

        except Exception as e:
            raise Custom_Exception(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTranformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))





