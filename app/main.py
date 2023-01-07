# Imports for server
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# App name
app = FastAPI(title="Loan Default Classifier for lending firm Bandora")

# defining base class for Loan to represent a data point for predictions 
class Loan(BaseModel):
    LanguageCode : object
    HomeOwnershipType : object
    Restructured : object
    IncomeTotal : float
    LiabilitiesTotal : float
    LoanDuration : float
    AppliedAmount : float
    Amount : float
    Interest : float
    EMI : float
    PreviousRepaymentsBeforeLoan : float
    MonthlyPaymentDay : float
    PrincipalPaymentsMade : float
    InterestAndPenaltyPaymentsMade : float
    PrincipalBalance : float
    InterestAndPenaltyBalance : float
    Bids : float
    Rating : object


@app.on_event("startup")
def load_ml_pipeline():
    # loading the machine learning pipeline from pickle .sav format
    global RFC_pipeline
    RFC_pipeline = pickle.load(open('app/ML_artifact/RFC_pipeline.sav', 'rb'))

# Defining the function for handling the prediction requests, it will be run by '/predict' endpoint of server
# and expects a Loan class datapoint for prediction
@app.post("/predict")
def predict(inference_request : Loan):
    # creating a pandas dataframe to be fed to RandomForestClassifier pipeline for prediction
    input_dictionary = {
            "LanguageCode" : inference_request.LanguageCode,
            "HomeOwnershipType": inference_request.HomeOwnershipType,
            "Restructured" : inference_request.Restructured,
            "IncomeTotal" : inference_request.IncomeTotal,
            "LiabilitiesTotal" : inference_request.LiabilitiesTotal,
            "LoanDuration" : inference_request.LoanDuration,
            "AppliedAmount" : inference_request.AppliedAmount,
            "Amount": inference_request.Amount,
            "Interest":inference_request.Interest,
            "EMI": inference_request.EMI,
            "PreviousRepaymentsBeforeLoan" : inference_request.PreviousRepaymentsBeforeLoan,
            "MonthlyPaymentDay" :inference_request.MonthlyPaymentDay,
            "PrincipalPaymentsMade" : inference_request.PrincipalPaymentsMade,
            "InterestAndPenaltyPaymentsMade" : inference_request.InterestAndPenaltyPaymentsMade,
            "PrincipalBalance" : inference_request.PrincipalBalance,
            "InterestAndPenaltyBalance" : inference_request.InterestAndPenaltyBalance,
            "Bids" : inference_request.Bids,
            "Rating" : inference_request.Rating
    }
    inference_request_Data = pd.DataFrame(input_dictionary,index=[0])
    prediction = RFC_pipeline.predict(inference_request_Data)
    
    # Returning prediction
    if prediction == 0:
        return {"Prediction": "Not Defaulted"}
    else:
        return {"Prediction": "Defaulted"}


