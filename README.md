# Support Ticket Classification & Prioritization using Machine Learning

## Overview of the Project

Real-world organizations receive hundreds, or even thousands of tickets every day, on customer support teams.

Manually classifying these tickets into **issue type** and **urgency** is error-prone, time-consuming, and delays resolution of critical problems.

This project builds an **NLP-based Machine Learning system** that will automatically:

• Categorizes support tickets based on predefined categories Predicts the priority level of each ticket - High/Medium/Low. The system serves as an **automatic triage mechanism** that helps businesses respond more quickly and in a more effective manner.
____

## Problem Statement

Support teams face three major challenges:
* Tickets are not categorized properly
* Urgent issues are delayed due to manual sorting
* Time is wasted on triage instead of problem resolution

### Solution

An ML-powered decision-support system that:

* Reads raw ticket text
* Classifies the ticket category
* Assigns an appropriate priority level
____

## Dataset

**Customer Support Ticket Dataset (Kaggle)**

https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset 

* 8,469 real-world-style support tickets
* Contains ticket subject, description, type, and priority labels

## Key Columns Used

* **Ticket Subject**
* **Ticket Description**
* **Ticket Type** (Category label)
* **Ticket Priority** (High / Medium / Low)
____

## Approach

### 1. Text Preprocessing

* Combined ticket subject and description
* Converted text to lowercase
* Removed punctuation and numbers
* Removed stopwords using NLTK

### 2. Feature Extraction

* Used **TF-IDF Vectorization**
* Included unigrams and bigrams
* Limited vocabulary size to reduce overfitting

### 3. Modeling

Two supervised ML models were trained:

| Task                           | Model               |
| ------------------------------ | ------------------- |
| Ticket Category Classification | Logistic Regression |
| Ticket Priority Prediction     | Logistic Regression |

### 4. Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Class-wise performance analysis
____

## Results & Insights

* The category classifier identifies most ticket types correctly.
* Priority prediction puts the most emphasis on **high-priority ticket recall**, which is an important part of business operations.
* Misclassifications mainly occur between semantically similar ticket categories.

## Business Interpretation

* Automation of ticket categorization greatly reduces the effort involved in manual triaging. 
* The early identification of high-priority tickets means the possibility of quicker escalation, hence better customer satisfaction. 
* It supports customer support operations that are scaling.
____

## Technologies Used

* Python
* Scikit-learn
* NLTK
* TF-IDF Vectorization
* Pandas & NumPy
* Matplotlib & Seaborn
* Joblib
____

## How to Run the Project

1. Clone the repository

```bash
git clone https://github.com/your-username/support-ticket-ml.git
cd Ticket-Classification-Project
```

2. Create virtual environment and install dependencies

```bash
pip install -r requirements.txt
```

3. Run the project

```bash
python main.py

____

## 💡 Business Impact

This system can be directly used in:

* SaaS customer support platforms
* IT service desks
* Internal help-desk automation
* Enterprise customer experience optimization

It reduces response time, minimizes backlog, and improves operational efficiency.
____

Author- **Shreeya**