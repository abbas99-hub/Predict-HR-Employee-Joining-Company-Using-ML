import sys
import pandas as pd
from src.exception import Custom_Exception
from src.utils import load_object
import os 

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            path=r"C:\Users\Admin\ML_Projects\Predict_HR_Employee_Joining_Company\Predict-HR-Employee-Joining-Company-Using-ML\src\components\artifacts"
            model_path=os.path.join(path, "model.pkl")
            preprocessor_path=os.path.join(path, "pre_processor.pkl")
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise Custom_Exception(e,sys)



class CustomData:
    def __init__(  self,
        slno:int,         
        candidate_ref:int,
        doj_extended:str,
        duration_to_accept_offer:int,
        notice_period:int,
        offered_band:str,
        percent_hike_expected_ctc:int,
        percent_hike_offered_ctc:int,
        percent_difference_ctc:int,
        joining_bonus:int,
        candidate_relocate_actual:str,
        gender:str,
        candidate_source:str,
        rex_in_yrs:int,
        lob:str,
        location:str,
        age:int):

        self.slno=slno
        self.gender = gender
        self.candidate_ref =candidate_ref
        self.doj_extended=doj_extended
        self.duration_to_accept_offer=duration_to_accept_offer
        self.notice_period=notice_period
        self.offered_band=offered_band
        self.percent_hike_expected_ctc=percent_hike_expected_ctc
        self.percent_hike_offered_ctc=percent_hike_offered_ctc
        self.percent_difference_ctc=percent_difference_ctc
        self.joining_bonus=joining_bonus
        self.candidate_relocate_actual=candidate_relocate_actual
        self.candidate_source=candidate_source
        self.rex_in_yrs=rex_in_yrs
        self.lob=lob
        self.location=location
        self.age=age
       
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'Candidate Ref': [self.candidate_ref],
        'DOJ Extended': [self.doj_extended],
        'Duration to accept offer': [self.duration_to_accept_offer],
        'Notice period': [self.notice_period],
        'Offered band': [self.offered_band],
        'Pecent hike expected in CTC': [self.percent_hike_expected_ctc],
        'Percent hike offered in CTC': [self.percent_hike_offered_ctc],
        'Percent difference CTC': [self.percent_difference_ctc],
        'Joining Bonus': [self.joining_bonus],
        'Candidate relocate actual': [self.candidate_relocate_actual],
        'Gender': [self.gender],
        'Candidate Source': [self.candidate_source],
        'Rex in Yrs': [self.rex_in_yrs],
        'LOB': [self.lob],
        'Location': [self.location],
        'Age': [self.age],
        'SLNO':[self.slno]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise Custom_Exception(e, sys)