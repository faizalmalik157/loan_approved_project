# 🏦 Loan Approval Prediction

A Machine Learning project that predicts whether a loan application is likely to be **approved or rejected** based on applicant and financial information.

The project was developed using **Python and Jupyter Notebook**, with data preprocessing and exploratory data analysis performed using **Pandas, NumPy, and Seaborn**. A **Random Forest Classifier** was used to build the machine learning model, and the final model was deployed as an interactive **Streamlit web application**.

## 🚀 Project Overview

Loan approval is an important decision for financial institutions. Traditionally, loan applications are evaluated using various applicant details such as income, credit history, loan amount, employment status, and other financial factors.

In this project, machine learning is used to learn patterns from historical loan application data and predict whether a new loan application should be **Approved** or **Rejected**.

### Project Workflow

```text
Dataset
   ↓
Data Cleaning & Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Random Forest Classifier
   ↓
Model Evaluation
   ↓
Streamlit Web App
   ↓
Loan Approval Prediction
```

## 🛠️ Technologies & Libraries Used

* **Python**
* **Jupyter Notebook**
* **NumPy** – Numerical operations
* **Pandas** – Data manipulation and preprocessing
* **Seaborn** – Data visualization
* **Matplotlib** – Data visualization
* **Scikit-learn** – Machine learning
* **Random Forest Classifier** – Prediction model
* **Streamlit** – Web application and model deployment
* **Git & GitHub** – Version control and project hosting

## 🤖 Machine Learning Model

### Random Forest Classifier

The project uses the **Random Forest Classifier** to predict loan approval.

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to make a final prediction.

Instead of relying on a single decision tree, Random Forest creates multiple trees and combines their predictions, which generally makes the model more robust and less prone to overfitting.

## 📊 Dataset

The dataset contains information about loan applicants, including features related to their personal and financial background.

Example features may include:

* Gender
* Married
* Dependents
* Education
* Self Employment
* Applicant Income
* Coapplicant Income
* Loan Amount
* Loan Term
* Credit History
* Property Area

### Target Variable

The target variable represents the loan decision:

* `Y` → Loan Approved
* `N` → Loan Rejected

## 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed using **Pandas, NumPy, Seaborn, and Matplotlib**.

The analysis included:

* Checking missing values
* Understanding dataset structure
* Statistical analysis
* Identifying categorical and numerical features
* Data visualization
* Understanding relationships between features
* Analyzing factors affecting loan approval

## 🧹 Data Preprocessing

Before training the model, the dataset was cleaned and prepared.

The preprocessing steps included:

1. Handling missing values
2. Converting categorical variables into numerical form
3. Selecting relevant features
4. Splitting the dataset into training and testing sets
5. Preparing the data for machine learning

## 📈 Model Training

The processed dataset was divided into training and testing data.

The Random Forest Classifier was then trained using the training dataset.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)

prediction = model.predict(X_test)
```

The trained model was evaluated on the test dataset to measure its prediction performance.

## 🌐 Streamlit Application

The trained machine learning model was integrated into a **Streamlit web application**.

The application allows users to enter loan applicant information and receive a prediction.

### Application Flow

```text
User enters applicant details
            ↓
Streamlit collects input
            ↓
Input preprocessing
            ↓
Random Forest Model
            ↓
Prediction
            ↓
Loan Approved / Loan Rejected
```

## 💻 Running the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/loan-approval-prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd loan-approval-prediction
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📁 Project Structure

```text
loan-approval-prediction/
│
├── data/
│   └── loan_data.csv
│
├── notebook/
│   └── loan_prediction.ipynb
│
├── model/
│   └── loan_model.pkl
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

> Adjust the filenames and folders according to your actual project structure.

## 📦 Requirements

The main dependencies used in this project are:

```text
numpy
pandas
seaborn
matplotlib
scikit-learn
streamlit
```

You can install them using:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn streamlit
```

## 🎯 Key Learning Outcomes

Through this project, I learned and practiced:

* Data cleaning and preprocessing
* Exploratory Data Analysis
* Data visualization
* Feature preparation
* Machine learning model training
* Random Forest classification
* Model evaluation
* Saving and loading ML models
* Building a Streamlit application
* Deploying a machine learning project
* Using Git and GitHub for project management

## 🔮 Future Improvements

Some possible improvements for the project include:

* Comparing Random Forest with other classification algorithms
* Hyperparameter tuning
* Improving model accuracy
* Adding more detailed model evaluation
* Adding feature importance visualization
* Improving the Streamlit UI
* Deploying the application online
* Adding probability/confidence scores to predictions

## 👨‍💻 Author

**Mohammad Faizal Malik**

B.Tech – Computer Engineering
Jamia Millia Islamia, New Delhi

---

⭐ If you find this project useful, consider giving the repository a star!
