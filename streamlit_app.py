import streamlit as st
import joblib
import numpy as np
import datetime

# Load models
random_forest_model = joblib.load('random_forest_model.joblib')
svr_model = joblib.load('svr_model.joblib')
linear_regression_model = joblib.load('linear_regression_model.joblib')

# Function to predict price using the models
def predict_price(model, data):
    data = np.array(list(data.values())).reshape(1, -1)
    return model.predict(data)[0]

# Create UI components for user input
st.title('House Price Prediction')

msZoning_code = st.selectbox('MSZoning', [0, 1, 2, 3])  # Assuming encoded categories
utility_code = st.selectbox('Utilities', [0, 1, 2])  # Assuming encoded categories
landSlope_code = st.selectbox('LandSlope', [0, 1, 2])  # Assuming encoded categories
buildingType_code = st.selectbox('BldgType', [0, 1, 2, 3, 4])  # Assuming encoded categories
overallQuality = st.slider('Overall Quality', 1, 10)
yearBuilt = st.date_input('Year Built', datetime.date(2000, 1, 1))
yearRemodAdd = st.date_input('Year Remodeled', datetime.date(2000, 1, 1))
totalBasmtSF = st.slider('Total Basement SF', 0, 5000)
totalRmsAbvGrd = st.slider('Total Rooms Above Ground', 1, 20)
floorSF = st.slider('1st Floor SF', 0, 5000)
grLiveArea = st.slider('GrLivArea', 0, 5000)
fullBath = st.slider('Full Bath', 0, 5)
kitchenQual_code = st.selectbox('Kitchen Quality', [0, 1, 2, 3])  # Assuming encoded categories
garageCars = st.slider('Garage Cars', 0, 5)
saleCondition_code = st.selectbox('Sale Condition', [0, 1, 2, 3, 4])  # Assuming encoded categories

# Prepare input data
data = {
    'MSZoning': msZoning_code,
    'Utilities': utility_code,
    'LandSlope': landSlope_code,
    'BldgType': buildingType_code,
    'OverallQual': overallQuality,
    'YearBuilt': yearBuilt.year,
    'YearRemodAdd': yearRemodAdd.year,
    'TotalBsmtSF': totalBasmtSF,
    'TotRmsAbvGrd': totalRmsAbvGrd,
    '1stFlrSF': floorSF,
    'GrLivArea': grLiveArea,
    'FullBath': fullBath,
    'KitchenQual': kitchenQual_code,
    'GarageCars': garageCars,
    'SaleCondition': saleCondition_code
}

# Display predictions
if st.button('Predict with Random Forest'):
    price = predict_price(random_forest_model, data)
    st.write(f'Predicted Price (Random Forest): ${price:,.2f}')

if st.button('Predict with SVR'):
    price = predict_price(svr_model, data)
    st.write(f'Predicted Price (SVR): ${price:,.2f}')

if st.button('Predict with Linear Regression'):
    price = predict_price(linear_regression_model, data)
    st.write(f'Predicted Price (Linear Regression): ${price:,.2f}')
