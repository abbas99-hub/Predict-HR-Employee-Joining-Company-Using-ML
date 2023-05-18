from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformationConfig,DataTranformation
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTranformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))



