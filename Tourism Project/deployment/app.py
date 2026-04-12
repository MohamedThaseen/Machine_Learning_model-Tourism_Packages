
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_model():
    # Download the model from your MODEL repo
    model_path = hf_hub_download(
        repo_id="Thaseen75/Tourism_Package_Prediction", 
        filename="Tourism_Package_Prediction_Model_v1.joblib"
    )
    return joblib.load(model_path)

model = load_model()

st.title("Tourism Package Prediction App")
st.subheader("Enter Customer & Package Details")


TypeofContact = st.selectbox("Type of Contact :",['Self Enquiry','Company Invited'])
Occupation = st.selectbox("Occupation : ", ['Salaried','Free Lancer','Small Business','Large Business'])
Gender = st.selectbox("Gender : ", ['Male','Female'])
ProductPitched = st.selectbox("Product Pitched : ", ['Deluxe','Basic','Standard','Super Deluxe','King'])
MaritalStatus = st.selectbox("Marital Status : ", ['Married','Single','Divorced'])
Designation = st.selectbox("Designation : ", ['Manager','Executive','Senior Manager','AVP','VP'])
CityTier = st.selectbox("City Tier : ", ['1','2','3'])
Passport = st.selectbox("Passport : ", ['0','1'])
OwnCar = st.selectbox("Own Car : ", ['0','1'])

Age = st.number_input("Age (Min: 18.0, Max: 61.0):", min_value=18.0, max_value=61.0)
DurationOfPitch = st.number_input("Duration Of Pitch(Min = 5.0, Max = 36.5):",Min = 5.0, Max = 36.5)
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting(Min = 1.0, Max = 4.5):",Min = 1.0, Max = 4.5)
PreferredPropertyStar = st.number_input("Preferred Property Star(Min = 3.0, Max = 5.0):",Min = 3.0, Max = 5.0)
NumberOfTrips = st.number_input("Number Of Trips(Min = 1.0, Max = 7.0):",Min = 1.0, Max = 7.0)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score(Min = 1.0, Max = 5.0):",Min = 1.0, Max = 5.0)
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting(Min = 0.0, Max = 3.0):",Min = 0.0, Max = 3.0)
MonthlyIncome = st.number_input("Monthly Income(Min = 13926.0, Max = 32126.0):",Min = 13926.0, Max = 32126.0)
NumberOfFollowups = st.number_input("Number Of Followups(Min = 1.5, Max = 5.5)",Min = 1.5, Max = 5.5)

input_data = pd.DataFrame([{

        'TypeofContact' : TypeofContact,
        'Occupation' : Occupation,
        'Gender' : Gender,
        'ProductPitched' : ProductPitched,
        'MaritalStatus' : MaritalStatus,
        'Designation' : Designation,
        'CityTier' : CityTier,
        'Passport' : Passport,
        'OwnCar' : OwnCar,
        'Age' : Age,
        'DurationOfPitch' : DurationOfPitch,
        'NumberOfPersonVisiting' : NumberOfPersonVisiting,
        'PreferredPropertyStar' : PreferredPropertyStar,
        'NumberOfTrips' : NumberOfTrips,
        'PitchSatisfactionScore' : PitchSatisfactionScore,
        'NumberOfChildrenVisiting' : NumberOfChildrenVisiting,
        'MonthlyIncome' : MonthlyIncome,
        'NumberOfFollowups' : NumberOfFollowups

    }])

if st.button("🔮 Predict"):
  Prediction = model.predict(input_data)[0]
  result = "Customer Will take the tourism package" if Prediction == 1 else "Customer Will not take the tourism package"
  st.info(f"The model has predicted that {result}")
  st.info(f"Predicted :{Prediction:.2f} ")
