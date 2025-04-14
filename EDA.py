import pandas as pd
from ydata_profiling import ProfileReport

data = pd.read_csv("diabetes_prediction_dataset.csv")

profile = ProfileReport(data, title= "Diabetes Profiling", explorative = True)
profile.to_file("report.html")

