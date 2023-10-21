import streamlit as st
import pandas as pd
import pickle

# Load your trained model
with open("lung_cancer.pkl", "rb") as file:  # Replace with your model filename
    model = pickle.load(file)

# Define the column names
column_names = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
               'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
               'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
               'SWALLOWING DIFFICULTY', 'CHEST PAIN']

# Create a dictionary to store user inputs
user_input = {}

# Sidebar with a side note
st.sidebar.title(" About this App")
st.sidebar.write("""Mr Aramide write something you want 
your users to know about this application here.
                 
                 """)

# Create a Streamlit app
st.title("Lung Cancer Risk Prediction App")
st.markdown("Predict your risk of lung cancer based on the provided information.")
st.image("lung cancer.jpg", use_column_width=True)

# Create input fields for user to enter values
for column in column_names:
    if column == 'GENDER':
        user_input[column] = st.selectbox(column, ['Female', 'Male'])
    else:
        user_input[column] = st.number_input(column, min_value=0, step=1)

# Create a button to make predictions
if st.button("Predict Now"):
    # Convert gender to numeric value
    user_input['GENDER'] = 1 if user_input['GENDER'] == 'Male' else 0

    # Create a dataframe from user input
    input_data = pd.DataFrame([user_input])

    # Make predictions with the model
    prediction = model.predict(input_data)

    # Map model output to "Yes" or "No"
    prediction_text = "Yes" if prediction[0] == 1 else "No"
    test_result = 'Opps! Your are having lung cancer' if prediction_text =='Yes' else 'Congrats Your are free from lung cancer'
    st.success(f"Prediction: {test_result}.")

# Add a footer with acknowledgments
st.markdown("""
---
*App created by Felix Sam Nanor.*
[GitHub Repository](https://github.com/Tech-Watt?tab=repositories)
""")
