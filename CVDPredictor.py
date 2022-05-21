import numpy as np
import pickle
import pandas as pd
from xgboost import XGBClassifier
import streamlit as st;
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler


print(xgb.__version__)


import pickle;


xgboostmodel= pickle.load(open('trainedxgboost.sav','rb'))
rfmodel= pickle.load(open('trainedxgboost.sav','rb'))
logisticmodel= pickle.load(open('trainedxgboost.sav','rb'))
svmmodel= pickle.load(open('trainedxgboost.sav','rb'))

testlabels=pickle.load(open('test20percentlabels.sav','rb'))
testfeatures=pickle.load(open('test20percentfeeature.sav','rb'))
scaler=pickle.load(open('scaler.sav','rb'))


def Predictor(model ,data) :
   prediction= model.predict(data)
   if (prediction==0 ):
      return ( str(prediction)  +'Doest not have CVD ')
   else :
     return ( str(prediction)  +'Has CVD ')







st.title("CVD Application -")
html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">CARDIO VASCULAR DISEASE PREDICTION MODEL </h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)
page = st.sidebar.selectbox('MACHINE LEARNING MODEL', ["XGBOOST", "RANDOM FOREST","LOGISTIC REGRESSION","SVM"])

st.sidebar.markdown("""---""")
st.sidebar.write("Created by  20042345-  O A Patrick")
st.sidebar.write("Supervised by  : Hisham ,Ignacio & Mahmoud  ")
st.sidebar.write("Improvement is on the way.......  ")
age=st.text_input('Enter the age in yrs',0)
weight=st.text_input('Enter the weight in kg',0)
height=st.text_input('Enter the height in ft',0)
ap_hi=st.text_input('Systolic Blood Pressure',0)
ap_lo=st.text_input('Distolic Blood Pressure',0)

gender = st.sidebar.slider("Gender", min_value=0,max_value=1,step=1)
smoke = st.sidebar.slider("Smoke", min_value=0,max_value=1,step=1)
alco = st.sidebar.slider("Alcohol", min_value=0,max_value=1,step=1)
active= st.sidebar.slider("Physical Activity", min_value=0,max_value=1,step=1)
gluc = st.sidebar.slider("Glocuse 1: normal, 2: above normal, 3: well above normal", min_value=1,max_value=3,step=1)
cholesterol = st.sidebar.slider("Cholestrol 1: normal, 2: above normal, 3: well above normal", min_value=1,max_value=3,step=1)


inputdata = {
'age':age,
'gender' :float(gender),
'height' :float(height),  
'weight' :float(weight),    
'ap_hi'  :float(ap_hi),
   
'ap_lo' :float(ap_lo),
'cholesterol':float(cholesterol ),
'gluc'  :float(gluc),
'smoke' :float(smoke) ,
'alco'  :float(alco),
'active':float(active)
           }



if st.button("Predict"):
      mydata= pd.DataFrame(inputdata , index=[0])
      mydatas=pd.DataFrame(inputdata , index=[0])
      mydata[mydata.columns.values]= scaler.transform(mydata)


      result=""
      model= xgboostmodel
      if    ( page == 'XGBOOST' ):             model= xgboostmodel
      elif  ( page == 'RANDOM FOREST' ):       model= rfmodel
      elif  ( page == 'LOGISTIC REGRESSION') : model= logisticmodel
      elif  ( page == 'SVM' ):                 model= svmmodel
  


      print(mydata)
      result=Predictor(model,mydata)
      print (result)

      st.success(f'The output from {page} is {result} ,{mydatas}  ==  \n {mydata}')
if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
