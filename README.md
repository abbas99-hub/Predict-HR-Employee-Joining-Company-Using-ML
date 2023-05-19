# Employee Attrition Prediction
This repository contains the code and instructions to build a machine learning model that predicts whether an employee will join a company. The project aims to help businesses identify potential employees who are likely to accept a job offer and join the organization. The model is trained on historical employee data and deployed using the Azure cloud platform.

## Business Understanding
The goal of this project is to develop a predictive model that can assist HR departments and hiring managers in determining the likelihood of a candidate accepting a job offer and joining the company. By leveraging machine learning techniques, we aim to provide insights into the factors that influence a candidate's decision and create a predictive model that can help prioritize and optimize the hiring process.

## Data Collection
To build the employee attrition prediction model, we need historical employee data. The following columns will be collected for analysis:

* SLNO: Serial number
* Candidate Ref: Candidate reference number
* DOJ Extended: Whether the candidate joined on an extended date
* Duration to accept offer: Time taken by the candidate to accept the job offer
* Notice period: Notice period required by the candidate before joining
* Offered band: Job band offered to the candidate
* Percent hike expected in CTC: Percentage increase in salary expected by the candidate
* Percent hike offered in CTC: Percentage increase in salary offered by the company
* Percent difference CTC: Difference between expected and offered salary as a percentage
* Joining Bonus: Whether the candidate received a joining bonus
* Candidate relocate actual: Whether the candidate relocated to join the company
* Gender: Gender of the candidate
* Candidate Source: Source from where the candidate was referred
* Rex in Yrs: Relevant experience of the candidate in years
* LOB: Line of business
* Location: Location of the candidate
* Age: Age of the candidate
* Status: Whether the candidate joined the company (target variable)

## Feature Engineering
Once the data is collected, we will perform feature engineering to preprocess and transform the raw data into meaningful features for model training. This step includes:

* Handling Missing Data: Deal with missing values in the dataset by either imputing them or removing the corresponding rows/columns.
* Encoding Categorical Variables: Convert categorical variables, such as candidate source and line of business, into numerical representations using techniques like one-hot encoding or label encoding.
* Feature Scaling: Normalize numerical features, such as age and experience, to ensure they have a similar scale and prevent dominance of certain features in the model.

## Model Training
We will train a machine learning model using the preprocessed data to predict employee attrition. The specific model architecture and algorithms used will depend on the nature of the prediction problem. Some potential models that can be explored include:

* Logistic Regression: Build a binary classification model that predicts whether a candidate will join the company.
* Random Forest: Develop an ensemble model that leverages multiple decision trees to make predictions.
* Gradient Boosting: Use boosting techniques to train an ensemble of weak prediction models that combine to make accurate predictions.

## Hyperparameter Tuning
To optimize the model's performance, we will perform hyperparameter tuning using both grid search and random search techniques. This step involves exploring different combinations of hyperparameters and evaluating the model's performance on a validation set. By tuning the hyperparameters, we can identify the best configuration that maximizes the model's predictive accuracy.

## Model Evaluation
After training the model with the tuned hyperparameters, we will evaluate its performance using suitable evaluation metrics such as accuracy, precision, recall, and F1-score. This step will help us assess how well the model predicts employee attrition and identify areas for improvement.

## Model Deployment using Azure Cloud
To make the employee attrition prediction model accessible, we will deploy it on the Azure cloud platform. The deployment process involves the following steps:

* Model Serialization: Serialize the trained model to a format compatible with Azure deployment.
* Azure Machine Learning: Set up an Azure Machine Learning workspace to manage the deployment process.
* Model Deployment: Deploy the serialized model as a web service using Azure Machine Learning service.
* API Development: Develop an API that allows users to interact with the deployed model and make predictions.
* Integration and Testing: Integrate the API with other components of the employee attrition prediction system and perform thorough testing to ensure its functionality and performance.
* Deployment Monitoring: Monitor the deployed model and API to track usage, performance metrics, and address any potential issues or errors.

## Usage
To use the employee attrition prediction system, follow the instructions below:

* Clone this repository: git clone <repository-url>
* Install the required dependencies: pip install -r requirements.txt
* Execute the data collection script or import the provided dataset: python collect_data.py or use data.csv.
* Perform feature engineering on the collected data: python feature_engineering.py
* Train the employee attrition prediction model: python train_model.py
* Perform hyperparameter tuning using grid search and random search: python hyperparameter_tuning.py
* Evaluate the model performance: python evaluate_model.py
* Serialize the trained model: python serialize_model.py
* Set up an Azure Machine Learning workspace: Follow the Azure documentation to create a workspace.
* Deploy the model as a web service using Azure Machine Learning service: Refer to the Azure Machine Learning documentation for detailed instructions.
* Access the deployed employee attrition prediction API and make requests to obtain predictions.

### Project Made By: Abbas Behrainwala

### Please feel free to contribute to this project by submitting pull requests or opening issues.
