
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

        # Define hyperparameters and their possible values
        self.param_dist = {
            'C': np.logspace(-6, 6, 14),  # Range of values for regularization strength
            'penalty': ['l1', 'l2'],     # Regularization types
            'solver': ['liblinear'],     # Specify solver for logistic regression
        }

    def test_model_accuracy(self):
        best_accuracy = 0

        # Number of iterations for randomized search
        num_iterations = 1000

        for i in range(num_iterations):
            # Define the number of features to select in each iteration
            num_features_to_select = random.randint(3, 8)
            # Randomly select a subset of features
            selected_features = np.random.choice(
                self.X_train_scaled.shape[1], num_features_to_select, replace=False)
            X_train_selected = self.X_train_scaled[:, selected_features]
            X_test_selected = self.X_test_scaled[:, selected_features]

            # Create Randomized Search with cross-validation
            random_search = RandomizedSearchCV(
                LogisticRegression(),
                param_distributions=self.param_dist,
                n_iter=20,                  # Number of random combinations to try
                cv=5,                       # Number of cross-validation folds
                scoring='accuracy',
                random_state=i             # Vary the random state for different iterations
            )

            # Fit the Randomized Search to your data
            random_search.fit(X_train_selected, self.y_train)

            # Use the best model for predictions
            best_model_iter = random_search.best_estimator_
            y_pred = best_model_iter.predict(X_test_selected)

            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)

            # Check if the current model has higher accuracy than the best so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
        # Assert that the achieved accuracy is greater than or equal to a specified threshold
        expected_accuracy_threshold = 0.8  # Adjust this threshold as needed
        self.assertGreaterEqual(best_accuracy, expected_accuracy_threshold)

        # Print the best accuracy
        print(f'Best Accuracy: {best_accuracy:.4f}')

        # Calculate ROC and AUC
        y_pred_proba = best_model_iter.predict_proba(X_test_selected)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        auc = roc_auc_score(self.y_test, y_pred_proba)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig('roc_curve.png')  # Save ROC curve figure