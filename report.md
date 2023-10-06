# Customer Churn Prediction

This project aims to predict customer churn (i.e., whether a customer will discontinue their subscription) based on various features in the Telco Customer Churn dataset.

## Methodology

### Data Preprocessing

- **Data Cleaning**: The dataset was checked for missing values, and any missing data was handled appropriately.
- **Feature Encoding**: Categorical features were encoded using label encoding.
- **Feature Scaling**: Standard scaling was applied to normalize the feature values.
- **Train-Test Split**: The data was split into training and testing sets with a 80-20 ratio.

### Model Selection

- **Logistic Regression**: A logistic regression model was chosen as the primary classification algorithm.
- **Hyperparameter Tuning**: A randomized search with cross-validation was used to find the best hyperparameters for the logistic regression model.

### Model Evaluation

- **Accuracy**: The model's performance was evaluated using accuracy as the primary metric.
- **Receiver Operating Characteristic (ROC) Curve**: The ROC curve was plotted to visualize the model's true positive rate and false positive rate.
- **Confusion Matrix**: A confusion matrix was generated to assess the model's performance in terms of true positives, true negatives, false positives, and false negatives.

## Results

The logistic regression model achieved an accuracy of approximately 81.16 % on the test set, indicating its ability to predict customer churn.

## Visualizations

- **ROC Curve**:

![ROC Curve](images/roc_curve.png)

- **Confusion Matrix**:

![Confusion Matrix](images/confusion_matrix.png)


## Repository Structure

- `data/`: Contains the dataset file (e.g., telco_customer_churn.csv).
- `src/`: Includes the source code files.
- `images/`: Stores saved visualizations.
- `src/test_logistic_regression.py`: The Python script for data preprocessing, model training, and evaluation.
- `report.ipynb`: A Jupyter Notebook report from what is done on google colab.
- `requirements.txt` : A text file consisting the required python packages
- `report.md`: A Markdown report (if available).
- `images/roc_curve.png`: The ROC curve image.
- `images/confusion_matrix.png`: The confusion matrix image.

## Usage

To run the project and reproduce the results, follow these steps:

1. Clone this repository:

   ```shell
   git clone https://github.com/MiladGhorbaniG/python-ai-developer.git
   cd python-ai-developer
   ```

2. Install the required dependencies:

    ```shell
    pip install -r requirements.txt
    ```

3. Execute the Python script for data preprocessing, model training, and evaluation:

    ```shell
    python src/test_logistic_regression.py
    ```

