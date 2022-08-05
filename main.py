from fastapi import FastAPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import joblib
from sklearn.preprocessing import StandardScaler
import json
import bz2file as bz2
import pickle
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/predictRFClassification')
def index ():
    def decompress_pickle(file):
        data = bz2.BZ2File(file, "rb")
        data = pickle.load(data)
        return data

    loaded_model = decompress_pickle("RFClass.pbz2")
    modified_data = pd.read_csv("cleaned.csv")
    modified_data = modified_data.drop(
        ["ListingCreationDate", "DateCreditPulled", "FirstRecordedCreditLine", "LoanOriginationDate",
         "LoanOriginationQuarter", "ListingNumber", "LoanNumber"], axis=1)
    modified_data = modified_data[
        ["ClosedDate", "LoanCurrentDaysDelinquent", "LoanMonthsSinceOrigination", "LP_CustomerPrincipalPayments",
         "LP_GrossPrincipalLoss", "LP_NetPrincipalLoss", "LP_CustomerPayments", "EmploymentStatus", "LP_ServiceFees",
         "LoanOriginalAmount", "Investors", "EstimatedReturn", "LP_InterestandFees", "MonthlyLoanPayment",
         "LP_CollectionFees", "EstimatedEffectiveYield", "EstimatedLoss", "Term", "BorrowerAPR",
         "LP_NonPrincipalRecoverypayments", "BorrowerRate", "ListingCategory (numeric)", "LenderYield",
         "CreditScoreRangeUpper", "OpenRevolvingMonthlyPayment", "ProsperScore", "CreditScoreRangeLower",
         "RevolvingCreditBalance", "ProsperRating (numeric)", "AvailableBankcardCredit", "EmploymentStatusDuration",
         "DebtToIncomeRatio", "StatedMonthlyIncome", "BankcardUtilization", "TotalCreditLinespast7years", "TotalTrades",
         "LoanStatus"]]

    y = modified_data["LoanStatus"]
    X = modified_data.drop(["LoanStatus"], axis=1)
    label_encoding_cols = ["EmploymentStatus"]
    for i in label_encoding_cols:
        X[i] = X[i].astype("category")
        X[i] = X[i].cat.codes

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    predictions = []
    predictions = loaded_model.predict(x_test)

    df = pd.DataFrame(predictions)
    df.index.names = ['Predictions:']
    df.to_csv("RFClassification.csv", index=False)

    jsonStr = df.to_json(orient='records')
    parsed = json.loads(jsonStr)

    result = loaded_model.score(x_test, y_test)
    print(result)

    return result, parsed

@app.get('/predictRFRegression')
def index ():
    def decompress_pickle(file):
        data = bz2.BZ2File(file, "rb")
        data = pickle.load(data)
        return data

    loaded_model = decompress_pickle("RFReg.pbz2")

    modified_data = pd.read_csv("cleaned.csv")
    modified_data = modified_data.drop(
        ["ListingCreationDate", "DateCreditPulled", "FirstRecordedCreditLine", "LoanOriginationDate",
         "LoanOriginationQuarter", "ListingNumber", "LoanNumber"], axis=1)
    modified_data = modified_data[
        ["ClosedDate", "LoanCurrentDaysDelinquent", "LoanMonthsSinceOrigination", "LP_CustomerPrincipalPayments",
         "LP_GrossPrincipalLoss", "LP_NetPrincipalLoss", "LP_CustomerPayments", "EmploymentStatus", "LP_ServiceFees",
         "LoanOriginalAmount", "Investors", "EstimatedReturn", "LP_InterestandFees", "MonthlyLoanPayment",
         "LP_CollectionFees", "EstimatedEffectiveYield", "EstimatedLoss", "Term", "BorrowerAPR",
         "LP_NonPrincipalRecoverypayments", "BorrowerRate", "ListingCategory (numeric)", "LenderYield",
         "CreditScoreRangeUpper", "OpenRevolvingMonthlyPayment", "ProsperScore", "CreditScoreRangeLower",
         "RevolvingCreditBalance", "ProsperRating (numeric)", "AvailableBankcardCredit", "EmploymentStatusDuration",
         "DebtToIncomeRatio", "StatedMonthlyIncome", "BankcardUtilization", "TotalCreditLinespast7years", "TotalTrades",
         "LoanStatus"]]

    y = modified_data["BorrowerRate"]
    X = modified_data.drop(["BorrowerRate"], axis=1)
    label_encoding_cols = ["EmploymentStatus", "LoanStatus"]
    for i in label_encoding_cols:
        X[i] = X[i].astype("category")
        X[i] = X[i].cat.codes

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    predictions = []
    predictions = loaded_model.predict(x_test)

    df = pd.DataFrame(predictions)
    df.index.names = ['Predictions:']
    df.to_csv("RFRegression.csv", index=False)

    jsonStr = df.to_json(orient='records')
    parsed = json.loads(jsonStr)

    result = loaded_model.score(x_test, y_test)
    print(result)

    return result, parsed

@app.get('/predictLRClassifier')
def index ():
    loaded_model = joblib.load("LRClassifier.joblib")
    data = pd.read_csv('cleaned.csv')
    y = data['LoanStatus']
    X = data.drop('LoanStatus', axis=1)
    cat_X = X.select_dtypes(include=('object'))
    X = X.drop(columns=cat_X.columns)
    for column in cat_X.columns:
        print(f'{column} : {len(cat_X[column].unique())}')
    cat_X = cat_X.drop(
        columns=['ListingCreationDate', 'DateCreditPulled', 'FirstRecordedCreditLine', 'LoanOriginationDate'])
    cat_X = pd.get_dummies(cat_X, drop_first=True)
    X.join(cat_X)
    sc = StandardScaler()
    scaled_X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)
    predictions = []
    predictions = loaded_model.predict(X_test)

    df = pd.DataFrame(predictions)
    df.index.names = ['Predictions:']
    df.to_csv("LRClassification.csv", index=False)

    jsonStr = df.to_json(orient='records')
    parsed = json.loads(jsonStr)

    result = loaded_model.score(X_test, y_test)
    print(result)

    return result, parsed

@app.get('/predictLRRegression')
def index ():
    loaded_model = joblib.load("LRRegression.joblib")
    data = pd.read_csv('cleaned.csv')
    y = data['LoanStatus']
    X = data.drop('LoanStatus', axis=1)
    cat_X = X.select_dtypes(include=('object'))
    X = X.drop(columns=cat_X.columns)
    for column in cat_X.columns:
        print(f'{column} : {len(cat_X[column].unique())}')
    cat_X = cat_X.drop(
        columns=['ListingCreationDate', 'DateCreditPulled', 'FirstRecordedCreditLine', 'LoanOriginationDate'])
    cat_X = pd.get_dummies(cat_X, drop_first=True)
    X.join(cat_X)
    sc = StandardScaler()
    scaled_X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)
    predictions = []
    predictions = loaded_model.predict(X_test)

    df = pd.DataFrame(predictions)
    df.index.names = ['Predictions:']
    df.to_csv("LRRegression.csv", index=False)

    jsonStr = df.to_json(orient='records')
    parsed = json.loads(jsonStr)

    result = loaded_model.score(X_test, y_test)
    print(result)

    return result, parsed



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

