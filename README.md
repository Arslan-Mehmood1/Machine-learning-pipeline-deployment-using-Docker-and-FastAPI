# Machine-learning-pipeline-deployment-using-Docker-and-FastAPI

This project is about deploying the trained machine learning pipeline using FastAPI and Docker. The trained machine learning pipeline which is to be deployed is taken from my repository [Credit-Risk-Analysis-for-european-peer-to-peer-lending-firm-Bandora](https://github.com/Arslan-Mehmood1/Credit-Risk-Analysis-for-european-peer-to-peer-lending-firm-Bandora).

The ml pipeline includes a RandomForestClassifier for classifying the loan borrowers as **defaulted / not-defaulted**.

## Building API using FastAPI framework
The API code must be in **'main.py'** file within a directory **'app'** according to FastAPI guidelines.

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

### Server Endpoint for Prediction
Finally, an endpoint on our server handles the **prediction requests** and return the value predicted by our deployed ml pipeline.

The endpoint is **server/predict** with a **POST** operation. 

Finally, a JSON response is returned containing the prediction

```python
# Defining the function for handling the prediction requests, it will be run by ```/predict``` endpoint of server
# and expects an instance inference request of Loan class to make prediction

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
```
### Server
As our API has been built, the Uvicorn Server can be use the API to serve the prediction requests. But for now, this server will be dockerized. And final predictions will be served by the Docker container.

## Dockerizing the Server
The Docker container will be run on localhost.
```
..
└── Base dir
    ├── app/
    │   ├── main.py (server code)
    │   └── ML_artifact (dir containing the RFC_pipeline.sav)
    ├── requirements.txt (Python dependencies)
    ├── loan-examples/ (loan examples to test the server)
    ├── README.md (this file)
    └── Dockerfile
```
## Creating the Dockerfile
Now in the base directory, a file is created ```Dockerfile``. The ```Dockerfile``` contain all the instructions required to build the docker image.

```Dockerfile
FROM frolvlad/alpine-miniconda3:python3.7

COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 80

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

### Base Image
The `FROM` instruction allows to use a pre-existing image as base of our new docker image, instead of writing our docker image from the scratch. This allows the software in pre-existing image to be available in our new docker image. 

In this case `frolvlad/alpine-miniconda3:python3.7` is used as base image.
- it contains python 3.7
- also contains an alpine version of linux, which is a distribution created to be very small in size.

Other existing images, can be used as base image of our new docker image, but size of those is a lot heavier. So using the one mentioned, as it a great image for required task.

### Installing Dependencies
Now our docker image has environment with python installed, so the dependencies required for serving the inference requests need to be installed in our docker image.

The dependencies are written in requirements.txt file in our base dir. This file needs to be copied in our docker image `COPY requirements.txt .` and then the dependencies are installed by `RUN pip install -r requirements.txt`

### Exposing the port
Our server will listen to inference requests on port 80.
```Dockerfile
EXPOSE 80
```

### Copying our App into Docker image
Our app should be inside the docker image.
```Dockerfile
COPY ./app /app
```

### Spinning up the server
Dockers are efficient at carrying out single task. When a docker container is run, the `CMD` commands get executed only once. This is the command which will start our server by specifying the `host` and `post`, when a docker container created from our docker image is started.
```Dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

## Build the Docker Image
Now in base dir, the Dockerfile is present. The Docker image is built using the `docker build` command:
```Dockerfile
docker build -t ml_pipeline:RFC
```
The `-t` flags is used for specifying the **name:tag** of docker image.

## Run the Docker Container
Now that the docker image is created. To run a docker container out of it:
```Dockerfile
docker run -p 80:80 ml_pipeline:RFC
```
The `-p 80:80` flag performs port mapping operations. The container and as well as local machine, has own set of ports. As our container is exposed on port 80, so it needs to be mapped to a port on local machine which is also 80.
<p align="center">
  <img src="/other/images/1.png">
</p>

## Make Inference Requests to Dockerized Server
Now that our server is listening on port 80, a `POST` request can be made for predicting the class of loan.

The requests should contain the data in `JSON` format.
```JSON
{
"LanguageCode": "estonian" ,
"AppliedAmount": 191.7349 ,
"Amount": 140.6057 ,
"Interest" : 25 , 
"LoanDuration" : 1, 
"EMI":3655.7482,
"HomeOwnershipType": "owner",
"IncomeTotal" : 1300.0,
"LiabilitiesTotal" : 0,
"MonthlyPaymentDay":15,
"Rating" : "f",
"Restructured" : "no",
"PrincipalPaymentsMade" : 140.6057,
"InterestAndPenaltyPaymentsMade" : 2.0227,
"PrincipalBalance" : 0,
"InterestAndPenaltyBalance" : 0,
"PreviousRepaymentsBeforeLoan" :258.6256,
"Bids" : 140.6057 
}
```
### FastAPI built-in Client
FastAPI has a built-in client to interact with the deployed server.
<p align="center">
  <img src="/other/images/3.png">
  <img src="/other/images/2.png">
</p>

### Using `curl` to send request
`curl` command can be used to send the inference request to deployed server.
```bash
curl -X POST http://localhost:80/predict \
    -d @./loan-examples/1.json \
    -H "Content-Type: application/json"
```
Three flags are used with `curl`:
`-X`: to specify the type of request like `POST`
`-d`: data to be sent with request
`-H`: header to specify the type of data sent with request

The directory `loan-examples` has 2 json files containing the loan samples for prediction, for testing the deployed dockerized server.
