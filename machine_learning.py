#Dataset: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/code
# Import necessary libraries
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import recall_score
# Load the dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")
# print(f"Original shape: {data.shape}")

#Remove duplicate
data = data.drop_duplicates()
# print(f"After removing duplicates: {data.shape}")

# Define target and features
target = "diabetes"
x = data.drop(target, axis = 1)
y = data[target]

# Split the data (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42, stratify=y)

#Preprocessing
##For numerical
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
##For norminal
nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False))
])
## For ordinal
smoking_values = ['No Info', 'never', 'former', 'not current', 'ever', 'current']
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[smoking_values]))
])
##Combine
preprocessor = ColumnTransformer([
    ("num_feature", num_transformer, ["age","hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]),
    ("nom_feature", nom_transformer, ["gender"]),
    ("ord_feature", ord_transformer, ["smoking_history"])
])
#Model
cls = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ("model", LogisticRegression())
])
# Train
cls.fit(x_train, y_train)
#Predict
y_predict = cls.predict(x_test)
#Evaluate
print(classification_report(y_test, y_predict))
print("Accuracy:", recall_score(y_test, y_predict))