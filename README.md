# Wells Fargo Credit Modeling Model Card

### Basic Information

* **Organization Developing Model**: GWU Wells Fargo Credit Modeling Team
* **Model Date**: May, 2023
* **Model Version**: 1.0
* **License**: MIT
* **Model Implementation Code**: 

### Intended Use
* **Primary intended uses**: 
* **Primary intended users**: 
* **Out-of-scope use cases**:

### Training Data

* Data Dictionary: 

| Name | Modeling Role | Data Type | Description|
| ---- | ------------- | --------- | ---------- |
| Mortgage | input | numerical | applicant’s mortgage size |
| Balance | input | numerical | average last 12 months credit card balance |
| Amount Past Due | input | numerical | the minimum required payment that was not applied to the account as of the last payment due date |
| Delinquency Status | input | ordinal | 0: current, 1: < 30-day delinquent, 2: 30–60-day delinquent, 3: 60-90 day and so on |
| Credit Inquiry | input | ordinal | number of credit inquiries in the last 12 months |
| Open Trade | input | ordinal | number of open credit accounts |
| Utilization | input | numerical | % credit utilization, the sum of all your balances, divided by the sum of your cards' credit limits |
| Gender | excluded | categorical | two kinds of gender (male and female) |
| Race | excluded | categorical | two kinds of race |
| Status | target | categorical | 0: default (should not be approved) and 1: non-default (should be approved). the 0/1 ratio is nearly 1:5 |

* **Source of training data**: 
* **How training data was randomly split into training and testing data**: 80% training and 20% testing
* **Number of rows in training and testing data**:
  * Training rows: 80,000
  * Testing rows: 20,000

### Model Details
* **Columns used as inputs in the final model**: 
* **Column(s) used as target(s) in the final model**: 'Status'
* **Type of model**: 
* **Software used to implement the model**: 
* **Version of the modeling software**: 
  * PiML version:
* **Hyperparameters or other settings of your model**:

### Quantitative Analysis

#### Correlation Heatmap

![Correlation Heatmap](https://github.com/tariaherbert/Wells-Fargo/blob/main/correlation%20heatmap.png)
* **The correlation heatmap demonstrates the strength of relationships between variables.** 
* **The heatmap demonstrates a moderately postive relationship between the following variables**:
   * Amount Past Due and Balance
   * Amount Past Due and Delinquency Status
   * Balance and Utilization
   * Credit Inquiry and Delinquency Status
   * Credit Inquiry and Open Trade
   * Delinquency Status and Open Trade
* **The heatmap demonstrates a moderately negative relationship between Status and Delinquency Status.**

#### Model Performance

| Rank |  Model  | Test ACC | Test AUC | Test F1 Score | Train ACC | Train AUC | Train F1 Score |
| ---- | ------- | -------- | -------- | ------------- | --------- | --------- | -------------- |
| 1 | ReLU-DNN | 0.6725 | 0.7354 | 0.7024 | 0.6725 | 0.7356 | 0.7030 |
| 2 | XGB2 | 0.6721 | 0.7312 | 0.7023 | 0.6755 | 0.7372 | 0.7065 |
| 3 | EBM | 0.6694 | 0.7310 | 0.7013 | 0.6736 | 0.7350 | 0.7060 |
| 4 | GAMI-Net | 0.6691 | 0.7280 | 0.7026 | 0.6701 | 0.7292 | 0.7045 |
| 5 | FIGS | 0.6681 | 0.7274 | 0.6926 | 0.6746 | 0.7371 | 0.6997 |
| 6 | GAM | 0.6670 | 0.7224 | 0.6956 | 0.6668 | 0.7229 | 0.6962 |
| 7 | Tree | 0.6654 | 0.7222 | 0.6976 | 0.6709 | 0.7271 | 0.7033 |
| 8 | GLM | 0.6521 | 0.7066 | 0.6920 | 0.6522 | 0.7077 | 0.6939 |

#### Model Robustness

![Model Robustness]()

#### Model Fairness

![Model Fairness]()
