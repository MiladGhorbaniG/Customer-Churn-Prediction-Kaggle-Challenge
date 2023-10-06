
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