import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt 
import pickle

# Load the dataset
df = pd.read_csv('survey lung cancer.csv')

# Exploring the data
# print(df.head())
# print(df.tail())
# print(df.columns)
# print(df.info())
# print(df.describe())
# print(df.dtypes)


# Convert 'GENDER' and 'LUNG_CANCER' to numerical values
df['GENDER'] = df['GENDER'].apply(lambda x: 0 if x == 'F' else 1)
df['LUNG_CANCER'] = df['LUNG_CANCER'].apply(lambda x: 1 if x == 'YES' else 0)

# Split the data into features (x) and labels (y)
x = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']
# print(x)
# print(y)


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Creating a data pipeline
preprocessing_step = [('scaler',StandardScaler())]
model_steps = [('classifier',RandomForestClassifier())]
pipeline = Pipeline(preprocessing_step+model_steps)
# training the model
pipeline.fit(x_train,y_train)
# making predictions on the test set
yhat = pipeline.predict(x_test)
print(yhat)

# Evaluating the model
accuracy = accuracy_score(y_test, yhat)
F1SCORE = f1_score(y_test,yhat)
RECALL  = recall_score(y_test,yhat)
PRECISION = precision_score(y_test,yhat)


print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {F1SCORE:.2f}")
print(f"Recall: {RECALL:.2f}")
print(f"Precision: {PRECISION:.2f}")


# Testing with new data
realdata = [[1, 69, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2]]
prediction = pipeline.predict(realdata)
print(f"Prediction for new data: {prediction[0]}")

# saving the model
model_filename = "lung_cancer.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(pipeline, file)