import pickle

# Load the model from the file
with open('lung_cancer.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Use the loaded model for predictions
new_data = [[1, 69, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2]]
predictions = loaded_model.predict(new_data)
print(F'Your output is {predictions[0]}')
