import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier


st.set_option('deprecation.showfileUploaderEncoding', False)
st.write("""
# Credit Default Prediction
""")
st.write('---')

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
def user_input_features():
    edu = st.sidebar.selectbox('EDUCATION',('School Graduate','University','High School','Others'))
    marr = st.sidebar.selectbox('MARRIAGE',('Married','Single','Others'))
    age = st.sidebar.slider('AGE', 16,100,25)
    limit = st.sidebar.slider('BALANCE LIMIT', 10000,1000000,100000)
    pay1 = st.sidebar.selectbox('PREVIOUS MONTH PAYMENT', ('-2','-1','0','1','2','3','4','5','6','7','8'))
    bill1 = st.sidebar.slider('BILL AMOUNT 1', -200000.00,1000000.00,0.00)
    bill2 = st.sidebar.slider('BILL AMOUNT 2', -200000.00,1000000.00,0.00)
    bill3 = st.sidebar.slider('BILL AMOUNT 3', -200000.00,1000000.00,0.00)
    bill4 = st.sidebar.slider('BILL AMOUNT 4', -200000.00,1000000.00,0.00)
    bill5 = st.sidebar.slider('BILL AMOUNT 5', -200000.00,1000000.00,0.00)
    bill6 = st.sidebar.slider('BILL AMOUNT 6', -200000.00,1000000.00,0.00)
    paya1 = st.sidebar.slider('PAY AMOUNT 1', 0.00,1000000.00,10000.00)
    paya2 = st.sidebar.slider('PAY AMOUNT 2', 0.00,1000000.00,10000.00)
    paya3 = st.sidebar.slider('PAY AMOUNT 3', 0.00,1000000.00,10000.00)
    paya4 = st.sidebar.slider('PAY AMOUNT 4', 0.00,1000000.00,10000.00)
    paya5 = st.sidebar.slider('PAY AMOUNT 5', 0.00,1000000.00,10000.00)
    paya6 = st.sidebar.slider('PAY AMOUNT 6', 0.00,1000000.00,10000.00)
    if edu=="School Graduate":
        educ = 1
    elif edu == "University":
        educ =2
    elif edu == "High School":
        educ = 3
    else:
        educ = 4
    if marr == "Married":
        marry =1
    elif marr == "Single":
        marry =2
    else:
        marry = 3
    data = {'LIMIT_BAL': limit,
            'EDUCATION': educ,
            'MARRIAGE': marry,
            'AGE': age,
            'PAY_1': pay1,
            'BILL_AMT1': bill1,
            'BILL_AMT2': bill2,
            'BILL_AMT3': bill3,
            'BILL_AMT4': bill4,
            'BILL_AMT5': bill5,
            'BILL_AMT6': bill6,
            'PAY_AMT1': paya1,
            'PAY_AMT2': paya2,
            'PAY_AMT3': paya3,
            'PAY_AMT4': paya4,
            'PAY_AMT5': paya5,
            'PAY_AMT6': paya6}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
credit = pd.read_csv('cleaned_appdata.csv')
credit_use = credit.drop(columns=['DEFAULT'])
df = pd.concat([input_df,credit_use],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
#encode = ['EDUCATION','MARRIAGE']
#for col in encode:
#    dummy = pd.get_dummies(df[col], prefix=col)
##    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')
st.write(df)
st.write('---')
# Reads in saved classification model
load_rf = joblib.load('default.pkl')

# Apply model to make predictions
prediction = load_rf.predict(df)
prediction_proba = load_rf.predict_proba(df)


st.subheader('Prediction')
status = np.array(['Not Default','Default'])
st.write(status[prediction])
st.write('---')
st.subheader('Prediction Probability')
st.write(prediction_proba)
st.write('---')
if all(prediction_proba[0]>=0.25):
    if all(prediction==0):
        st.subheader('THE ACCOUNT SHOULD BE CONSIDERED FOR **CREDIT COUNSELING**')
