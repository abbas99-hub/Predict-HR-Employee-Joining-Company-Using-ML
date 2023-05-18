import numpy as np
import pandas as pd
from  src.exception import Custom_Exception
import os
from src.logger import logging
from src.utils import save_object
import sys
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder,MinMaxScaler
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','pre_processor.pkl')

class DataTranformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_tranformer_object(self):
        try:
            numerical_features=['SLNO', 'Candidate Ref', 'Duration to accept offer', 'Notice period', 
                                'Pecent hike expected in CTC', 'Percent hike offered in CTC', 
                                'Percent difference CTC', 'Rex in Yrs', 'Age']
            
            categorical_features=['DOJ Extended', 'Offered band', 'Joining Bonus', 
                                  'Candidate relocate actual', 'Gender', 'Candidate Source', 
                                  'LOB', 'Location']
        
            num_pipeline=Pipeline(
            steps=[
               ( "imputer",SimpleImputer(strategy='median')),
               ("scaler",MinMaxScaler())
            ]
        )

            cat_pipeline=Pipeline(
             steps=[
               ( "imputer",SimpleImputer(strategy='most_frequent')),
               ("one_hot_encoder",OneHotEncoder())
            ]
        )    
            preprocessor=ColumnTransformer(
            [
                ("num_pipeline",num_pipeline,numerical_features),
                ("cat_pipeline",cat_pipeline,categorical_features)

            ]
        )
            return preprocessor
            
        except Exception as e:
            raise Custom_Exception(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Reading of Train and Test data has been done successfully.')
            preprocessing_obj = self.get_data_tranformer_object()
            target_column = "Status"
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            train_df[target_column] = train_df[target_column].replace({'Joined': 1, 'Not Joined': 0})
            test_df[target_column] = test_df[target_column].replace({'Joined': 1, 'Not Joined': 0})
            target_feature_train_df = train_df[target_column]
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]
            logging.info('Applying pre-processing object on training and testing dataframe')
            
            
            # Fit the preprocessor on training data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            # Apply the preprocessor on testing data
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Concatenate the target feature to the transformed training and testing arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info('Data Transformation obj created successfully')
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path


        except Exception as e:
            raise Custom_Exception(e, sys)
