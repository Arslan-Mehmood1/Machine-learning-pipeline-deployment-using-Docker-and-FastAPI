# Machine-learning-pipeline-deployment-using-Docker-and-FastAPI

This project is about deploying the trained machine learning pipeline using FastAPI and Docker. The trained machine learning pipeline which is to be deployed is taken from my repository [Credit-Risk-Analysis-for-european-peer-to-peer-lending-firm-Bandora](https://github.com/Arslan-Mehmood1/Credit-Risk-Analysis-for-european-peer-to-peer-lending-firm-Bandora).

## Server
The server code must be in **'main.py'** file within a directory **'app'** according to FastAPI guidelines.

Import the required packages
```python
# Imports for server
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# App name
app = FastAPI(title="Loan Default Classifier for lending firm Bandora")
```

### Representing the loan data point
To represent a sample of loan details along with the data type of each atttribute, a class needs to be defined using the ```BaseModel``` from the pydantic library.
```python
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
```

### Loading the trained Machine learning pipeline
The trained machine learning pipeline needs to be loaded into memory, so it can be used for predictions in future.

One way is to load the machine learning pipeline during the startup of our ```Server```. To do this, the function needs to be decorated with ```@app.on_event("startup")```. This decorator ensures that the function loading the ml pipeline is triggered right when the Server starts.

The ml pipeline is stored in `app/ML_artifact' directory.

```python
@app.on_event("startup")
def load_ml_pipeline():
    # loading the machine learning pipeline from pickle .sav format
    global RFC_pipeline
    RFC_pipeline = pickle.load(open('app/ML_artifact/RFC_pipeline.sav', 'rb'))
```







