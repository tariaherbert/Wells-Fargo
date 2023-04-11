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

### Exploratory Data Analysis

#### Correlation Heatmap

![Correlation Heatmap](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/correlation%20heatmap.png)
* **The correlation heatmap demonstrates the strength of relationships between variables.** 
* **The heatmap demonstrates a moderately postive relationship between the following variables**:
   * Amount Past Due and Balance
   * Amount Past Due and Delinquency Status
   * Balance and Utilization
   * Credit Inquiry and Delinquency Status
   * Credit Inquiry and Open Trade
   * Delinquency Status and Open Trade
* **The heatmap demonstrates a moderately negative relationship between Status and Delinquency Status.**

### Model Performance

| Rank |   Model | Test ACC | Test AUC | Test F1 Score | Train ACC | Train AUC | Train F1 Score |
| ---- | ------- | -------- | -------- | ------------- | --------- | --------- | -------------- |
| 1 | ReLU-DNN | 0.6725 | 0.7354 | 0.7024 | 0.6725 | 0.7356 | 0.7030 |
| 2 | XGB2 | 0.6721 | 0.7312 | 0.7023 | 0.6755 | 0.7372 | 0.7065 |
| 3 | EBM | 0.6694 | 0.7310 | 0.7013 | 0.6736 | 0.7350 | 0.7060 |
| 4 | GAMI-Net | 0.6691 | 0.7280 | 0.7026 | 0.6701 | 0.7292 | 0.7045 |
| 5 | FIGS | 0.6681 | 0.7274 | 0.6926 | 0.6746 | 0.7371 | 0.6997 |
| 6 | GAM | 0.6670 | 0.7224 | 0.6956 | 0.6668 | 0.7229 | 0.6962 |
| 7 | Tree | 0.6654 | 0.7222 | 0.6976 | 0.6709 | 0.7271 | 0.7033 |
| 8 | GLM | 0.6521 | 0.7066 | 0.6920 | 0.6522 | 0.7077 | 0.6939 |
* **Based on the test ACC metric results, out of the 8 models the best performing models are ReLU-DNN, XGB2, EBM, and GAMI-Net. Further analysis will be continued with the following models: ReLU-DNN, XGB2, and GAMI-Net. GAMI-Net was selected over EBM because XGB2 and EBM operate similarly.**

#### Hyperparameter Tuning

### Explainability

#### Global Explainability: XGB2

* **Permutation Feature Importance (PFI)**:
   
   ![XGB2 PFI](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20PFI.png)
   * Amount Past Due is the most influencial feature in the model
   * Utilization, Mortgage, and Balance have a significant impact on the response variable (Status)

* **Partial Dependency Plot (PDP) and Accumulated Local Effects (ALE)**:
   
   ![Amount Past Due](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20amount%20past%20due%20plots.png)
   * There is a negative relationship between Status approval and Amount Past Due. As Amount Past Due increases, the probability of that individual being approved for credit decreases and denied for credit increases.
   
   ![Balance](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20balance%20plots.png)
   * There is a positive relationship between Status approval and Balance. As Balance increases, the probability of that individual being approved for credit increases and denied for credit decreases.
   
   ![Credit Inquiry](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20credit%20inquiry%20plots.png)
   * There is a negative relationship between Status approval and Credit Inquiry. As Credit Inquiry increases, the probability of that individual being approved for credit decreases and denied for credit increases.
   
   ![Delinquency Status](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20delinquency%20status%20plots.png)
   * There is a negative relationship between Status approval and Delinquency Status. As Delinquency Status increases, the probability of that individual being approved for credit decreases and denied for credit increases.
   
   ![Mortgage](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20mortgage%20plots.png)
   * There is a positive relationship between Status approval and Mortgage. As Mortgage increases, the probability of that individual being approved for credit increases and denied for credit decreases.
   
   ![Open Trade](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20open%20trade%20plots.png)
   * There is a negative relationship between Status approval and Open Trade. As Open Trade increases, the probability of that individual being approved for credit decreases and denied for credit increases.
   
   ![Utilization](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20utilization%20plots.png)
   * There is a negative relationship between Status approval and Utilization. As Utilization increases, the probability of that individual being approved for credit decreases and denied for credit increases.

#### Local Explainability: XGB2

* **LIME and SHAP**:
   ![Sample 0](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20sample%200%20plots.png)
   ![Sample 5](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20sample%205%20plots.png)
   ![Sample 50](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20sample%2050%20plots.png)
   ![Sample 100](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20sample%20100%20plots.png)

### Interpretability

#### Global Interpretability: XGB2

#### Local Interpretability: XGB2

### Model Robustness

![All Features](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/model%20robustness%20all%20features.png)
![Amount Past Due](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/model%20robustness%20amount%20past%20due.png)
![Credit Inquiry](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/model%20robustness%20credit%20inquiry.png)
![Delinquency Status](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/model%20robustness%20delinquincy%20status.png)
![Mortgage](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/model%20robustness%20mortgage.png)
![Open Trade](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/model%20robustness%20open%20trade.png)
![Utilization](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/model%20robustness%20utilization.png)
* **The model robustness output indicates that out of the three models, GAMI-Net is the most robust model. The GAMI-Net model reacts well to the imputation of noise into the data. The XGB2 model has the highest accuracy associated with 0 pertubation (noise), but drops rapidly once noise is inputed into the data. After analyzing specific features, the XGB2 model responds the best to noise imputation for all features except Amount Past Due.**

### Model Fairness

![Model Fairness]()
