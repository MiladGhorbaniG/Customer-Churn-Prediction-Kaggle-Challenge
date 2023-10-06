
import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import joblib
import random
import matplotlib.pyplot as plt
import seaborn as sns
class TestLogisticRegression(unittest.TestCase):

    def setUp(self):
        # Load the dataset
        self.data = pd.read_csv('/content/WA_Fn-UseC_-Telco-Customer-Churn.csv')

        # Define target and relevant columns
        self.target_col = 'Churn'
        all_columns = self.data.columns.tolist()
        all_columns.remove(self.target_col)  # Exclude target column
        self.X = self.data[all_columns]
        self.y = self.data[self.target_col]

        # Encode categorical columns
        label_encoder = LabelEncoder()
        for col in self.X.select_dtypes(include=['object']).columns:
            self.X[col] = label_encoder.fit_transform(self.X[col])

        # Convert 'No' to 0 and 'Yes' to 1 in y
        self.y = self.y.map({'No': 0, 'Yes': 1})

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

        # Data Scaling
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)