import requests

# Prompt the user for input
age = int(input("Enter age: "))
hypertension = int(input("Hypertension (1 for Yes, 0 for No): "))
heart_disease = int(input("Heart disease (1 for Yes, 0 for No): "))
bmi = float(input("Enter BMI: "))
HbA1c_level = float(input("Enter HbA1c level: "))
blood_glucose_level = float(input("Enter blood glucose level: "))
gender = input("Enter gender (Male/Female/Other): ")
smoking_history = input("Enter smoking history (never/former/current/not current/ever/No Info): ")

# Prepare the data to be sent to the API
data = {
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "bmi": bmi,
    "HbA1c_level": HbA1c_level,
    "blood_glucose_level": blood_glucose_level,
    "gender": gender,
    "smoking_history": smoking_history
}


url = "https://diabetes-project-qbh4.onrender.com/predict"

response = requests.post(url, json=data)

print(response.json())
