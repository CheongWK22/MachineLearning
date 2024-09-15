import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load models, scaler, and feature names
loaded_random_forest = joblib.load('random_forest_model.joblib')
loaded_lin_reg = joblib.load('linear_regression_model.joblib')
column_names = joblib.load('saved_feature_names.pkl')
scaler = joblib.load('scaler.pkl')

# Load the dataset
df = pd.read_csv('train15.csv')

# Display the dataset
st.title('House Price Prediction')
with st.expander('Data'):
    st.write(df)

# Create the form for input
with st.sidebar:
    st.header('Input features')

    # Create input fields
    overallQuality = st.slider("Overall Material and Finish Quality", 1, 10, 5)
    yearBuilt = st.slider("Year Built", 1900, 2023, 2000)
    yearRemodAdd = st.slider("Year Remodeled", 1900, 2023, 2000)
    totalBasmtSF = st.slider("Total Basement Area (sqft)", 0, 5000, 500)
    firstFlrSF = st.slider("First Floor Area (sqft)", 0, 5000, 1000)
    grLivArea = st.slider("Above Ground Living Area (sqft)", 0, 5000, 1500)
    fullBath = st.slider("Number of Full Bathrooms", 0, 5, 2)
    totRmsAbvGrd = st.slider("Total Rooms Above Grade", 0, 15, 6)
    garageCars = st.slider("Garage Capacity (cars)", 0, 5, 2)

    # Categorical fields (you can add mapping logic similar to your original code)
    msZoning = st.selectbox('MS Zoning', ['RL', 'RM', 'C (all)', 'FV', 'RH'])
    utilities = st.selectbox('Utilities', ['AllPub', 'NoSewr'])
    bldgType = st.selectbox('Building Type', ['1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI'])
    kitchenQual = st.selectbox('Kitchen Quality', ['Ex', 'Gd', 'TA', 'Fa'])
    saleCondition = st.selectbox('Sale Condition', ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'])
    landSlope = st.selectbox('Land Slope', ['Gtl', 'Mod', 'Sev'])

# Create a dictionary of input features
data = {
    'OverallQual': overallQuality,
    'YearBuilt': yearBuilt,
    'YearRemodAdd': yearRemodAdd,
    'TotalBsmtSF': totalBasmtSF,
    '1stFlrSF': firstFlrSF,
    'GrLivArea': grLivArea,
    'FullBath': fullBath,
    'TotRmsAbvGrd': totRmsAbvGrd,
    'GarageCars': garageCars,
    'MSZoning': msZoning,
    'Utilities': utilities,
    'BldgType': bldgType,
    'KitchenQual': kitchenQual,
    'SaleCondition': saleCondition,
    'LandSlope': landSlope
}

# Convert the input into a DataFrame
input_df = pd.DataFrame(data, index=[0])

# Perform one-hot encoding for categorical columns
input_data_preprocessed = pd.get_dummies(input_df)

# Ensure input_data_preprocessed has the same columns as the training data
input_data_preprocessed = input_data_preprocessed.reindex(columns=column_names, fill_value=0)

# Scale numeric features
numeric_cols = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars']
input_data_preprocessed[numeric_cols] = scaler.transform(input_data_preprocessed[numeric_cols])

# Prediction using the loaded model
st.write("## Prediction Results")
if st.button('Predict'):
    lin_reg_pred = loaded_lin_reg.predict(input_data_preprocessed)
    st.write(f"**Linear Regression Prediction: ${lin_reg_pred[0]:,.2f}**")
