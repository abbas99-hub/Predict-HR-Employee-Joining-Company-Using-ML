import streamlit as st
import pandas as pd
import pickle
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformationConfig,DataTranformation
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
from src.pipeline.test_pipeline import CustomData, PredictPipeline
import random

# Load the trained model
model = pickle.load(open(r'C:\Users\Admin\ML_Projects\Predict_HR_Employee_Joining_Company\Predict-HR-Employee-Joining-Company-Using-ML\src\components\artifacts\model.pkl', 'rb'))

# Define the Streamlit app
def main():
    # Set the app title
    st.title("Employee Joining Predictor")

    # Add input fields for the features
    candidate_ref = st.number_input("Candidate (Unique reference number)")
    doj_extended = st.radio("DOJ Extended (Date of joining asked by candidate or not)", ["Yes", "No"])
    duration_to_accept_offer = st.number_input("Duration to accept offer (in days)")
    notice_period = st.number_input("Notice period served before candidate can join the company (in days)")
    offered_band = st.selectbox("Offered band", ["E1", "E2", "E3"])
    percent_hike_expected_ctc = st.number_input("Percent hike expected in CTC")
    percent_hike_offered_ctc = st.number_input("Percent hike offered in CTC")
    percent_difference_ctc = st.number_input("Percent difference CTC")
    joining_bonus = st.radio("Joining bonus given or not", ["Yes", "No"])
    candidate_relocate_actual = st.radio("Candidates have to relocate or not", ["Yes", "No"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    candidate_source = st.selectbox("Candidate Source", ["Employee Referral", "Agency", "Direct"])
    rex_in_yrs = st.number_input("Relevant years of experience")
    lob = st.selectbox("Line of business for which offer was rolled out", ["INFRA","ERS","Healthcare","BFSI","CSMP","ETS","AXON","EAS","MMS","Others"])
    location = st.selectbox("Company location for which offer was rolled out", ["Noida","Chennai","Bangalore","Gurgaon","Hyderabad","Kolkata","Cochin","Pune","Mumbai","Ahemadabad","Others"])
    age = st.number_input("Age")
    slno= random.randint(1, 100)
    
    # Create a dictionary with the input features
    input_data = CustomData(
        slno,
        candidate_ref,
        doj_extended,
        duration_to_accept_offer,
        notice_period,
        offered_band,
        percent_hike_expected_ctc,
        percent_hike_offered_ctc,
        percent_difference_ctc,
        joining_bonus,
        candidate_relocate_actual,
        gender,
        candidate_source,
        rex_in_yrs,
        lob,
        location,
        age
    )

    
    pred_df=input_data.get_data_as_data_frame()
    predict_pipeline=PredictPipeline()
    prediction=predict_pipeline.predict(pred_df)

    # Create a button for prediction
    if st.button("Predict"):
        pred_df = input_data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)

        # Display the prediction result
        if prediction[0] == 1:
            st.subheader("Prediction Result")
            st.write("The employee is likely to join the company")
        else:
            st.subheader("Prediction Result")
            st.write("The employee is not likely to join the company")


# Run the Streamlit app
if __name__ == '__main__':
    main()