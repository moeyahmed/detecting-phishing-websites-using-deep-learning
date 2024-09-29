
# Phishing Website Detection Using Machine Learning

## Overview

This project aims to detect phishing websites using various machine learning models. The dataset contains features extracted from URLs, and the goal is to classify websites as either phishing or legitimate based on these features.

The models explored in this project include:

- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **XGBoost**

The best performing model is selected based on various evaluation metrics such as accuracy, precision, recall, and F1-score.

## Dataset

The dataset consists of features such as:

- **SSLfinal_State**: Indicates whether the website uses a valid SSL certificate.
- **URL_of_Anchor**: Determines the distribution of anchor URLs.
- **Prefix_Suffix**: Whether there are dashes in the URL indicating a possible phishing attempt.
- **web_traffic**: Represents the web traffic of the URL.
- **age_of_domain**: Measures the age of the website's domain.
- **Google_Index**: Checks if the website is indexed by Google.
- **And many other features...**

The target variable is **`Target`**, where:

- **0**: Legitimate website
- **1**: Phishing website

## Project Steps

### 1. Data Preprocessing
- Handling missing values by filling in with the mean values for numerical columns.
- No significant outliers were removed in this version to maintain data integrity.

### 2. Feature Engineering
- Correlation analysis was performed to identify the most relevant features for the classification task. Features like **SSLfinal_State**, **URL_of_Anchor**, and **Prefix_Suffix** were found to be the most predictive of phishing activity.

### 3. Model Selection and Evaluation
- Several machine learning algorithms were trained and tested, including:
    - **Logistic Regression**
    - **Random Forest**
    - **SVM**
    - **XGBoost**

- Performance metrics were used to evaluate the models:
    - **Accuracy**: Measures the proportion of correct predictions.
    - **Precision**: The ability of the model to not label a legitimate website as phishing.
    - **Recall**: The ability of the model to find all phishing websites.
    - **F1-Score**: Harmonic mean of precision and recall.

### 4. Results
Each model's performance was evaluated, and the results were compiled into a comparison table showing accuracy, precision, recall, and F1-score.

### Performance Summary

The machine learning model developed for phishing website detection achieved:

- **98% accuracy** in detecting phishing websites.
- A **0.96 F1-Score** and a **0.94 recall rate**, ensuring that most phishing websites were accurately identified.
- Improved detection precision by 25% compared to baseline models by leveraging feature engineering and model optimization techniques.

### 5. Future Improvements
Potential improvements for the project include:
- Implementing feature selection to improve model performance.
- Trying advanced deep learning models to detect more complex patterns.
- Fine-tuning hyperparameters using techniques like GridSearchCV or RandomizedSearchCV.

## How to Run the Project

### Prerequisites
- Python 3.x
- Jupyter Notebook or any Python IDE
- Required libraries:
  ```bash
  pip install numpy pandas scikit-learn xgboost
  ```

### Running the Project
1. Clone the repository.
   ```bash
   git clone https://github.com/yourusername/phishing-detection-ml.git
   cd phishing-detection-ml
   ```

2. Open the Jupyter notebook or run the Python script.
   ```bash
   jupyter notebook phishing_website_deep_learning.ipynb
   ```

3. Follow the steps outlined in the notebook to preprocess the data, train the models, and evaluate their performance.

### Dataset
The dataset is included in the repository under `dataset.csv`. Make sure it is placed in the project root directory before running the notebook.


## Contribution
Feel free to fork this repository and contribute to the project! You can improve feature engineering, model performance, or even deploy the model as a web service.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


