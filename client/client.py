import requests

data = {
  "age": 58,
  "hypertension": 1,
  "heart_disease": 1,
  "bmi": 35.7,
  "HbA1c_level": 8.9,
  "blood_glucose_level": 230,
  "gender": "Male",
  "smoking_history": "former"
}


url = "https://diabetes-project-qbh4.onrender.com/predict"

response = requests.post(url, json=data)

print(response.json())
