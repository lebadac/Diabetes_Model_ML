# Import necessary libraries
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from mlflow.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from lazypredict.Supervised import LazyClassifier
# Load the dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")

# Remove duplicate rows
data = data.drop_duplicates()

# Define target and features
target = "diabetes"
x = data.drop(target, axis = 1)
y = data[target]

# Split the data (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42, stratify=y)
print("Target value counts in test set:")
print(y_test.value_counts())

# Preprocessing for numerical features
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocessing for nominal features
nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False))
])

# Preprocessing for ordinal features
smoking_values = ['No Info', 'never', 'former', 'not current', 'ever', 'current']
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[smoking_values]))
])

# Combine all transformers
preprocessor = ColumnTransformer([
    ("num_feature", num_transformer, ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]),
    ("nom_feature", nom_transformer, ["gender"]),
    ("ord_feature", ord_transformer, ["smoking_history"])
])

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)
sm = SMOTE(random_state=42)
x_train, y_train = sm.fit_resample(x_train, y_train)

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LazyClassifier(
    verbose=1,
    ignore_warnings=True,
    custom_metric=recall_score,
    classifiers=[
        LogisticRegression,
        RandomForestClassifier,
        DecisionTreeClassifier,
        AdaBoostClassifier,
        GradientBoostingClassifier,
        XGBClassifier,
        GaussianNB,
        KNeighborsClassifier,
        LinearDiscriminantAnalysis
    ]
)


models, predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)
