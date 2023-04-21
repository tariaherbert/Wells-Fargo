# Wells Fargo Credit Modeling Model Card

## Basic Information

* **Organization Developing Model**: GWU Wells Fargo Credit Modeling Team
* **Model Date**: May, 2023
* **Model Version**: 1.0
* **License**: Apache 2.0
* **Model Implementation Code**: 

## Intended Use
* **Primary intended uses**: This model is an example of an end-to-end credit modeling process that is intended to be used as a guide for current and future credit modelers.
* **Primary intended users**: Wells Fargo Team, Patrick Hall, and GWU Students in DNSC 6317
* **Out-of-scope use cases**: Any use beyond an educational example is out-of-scope.

## Training Data

* **Data Dictionary**: 

| Name | Modeling Role | Data Type | Description|
| ---- | ------------- | --------- | ---------- |
| Mortgage | input | numerical | applicant’s mortgage size |
| Balance | input | numerical | average last 12 months credit card balance |
| Amount Past Due | input | numerical | the minimum required payment that was not applied to the account as of the last payment due date |
| Delinquency Status | input | ordinal | 0: current, 1: < 30-day delinquent, 2: 30–60-day delinquent, 3: 60-90 day and so on |
| Credit Inquiry | input | ordinal | number of credit inquiries in the last 12 months |
| Open Trade | input | ordinal | number of open credit accounts |
| Utilization | input | numerical | % credit utilization, the sum of all your balances, divided by the sum of your cards' credit limits |
| Gender | excluded | categorical | two kinds of gender (0: protected group, 1: reference group) |
| Race | excluded | categorical | two kinds of race (0: protected group, 1: reference group)|
| Status | target | categorical | 0: default (should not be approved) and 1: non-default (should be approved). the 0/1 ratio is nearly 1:5 |

* **Source of training data**: 
* **How training data was randomly split into training and testing data**: 80% training and 20% testing
* **Number of rows in training and testing data**:
  * Training rows: 80,000
  * Testing rows: 20,000

## Model Details
* **Columns used as inputs in the final model**: 
* **Column used as target in the final model**: 'Status'
* **Type of final model**: 
* **Software used to implement the models**: Numpy, PiML, Python, XGBoost
* **Version of the modeling software**: 
   * Numpy version: 1.23.5
   * PiML version: 4.3
   * Python version: 3.9.16
   * XGBoost version: 1.7.2
* **Hyperparameters or other settings of the final model**:

## Exploratory Data Analysis

### Correlation Heatmap

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

### Feature Importance

![Feature Importance](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/feature%20importance.png)
* **The feature importance plot displays how useful and often a model finds each feature when making accurate predictions about the target variable. Amount Past Due and Utilization are used the most in the machine learning models. Open Trade was removed from the models because it has no influence when making prediction about the target variable (Status).** 

## Model Performance

| Rank | Model | Test ACC | Test AUC | Test F1 Score | Train ACC | Train AUC | Train F1 Score |
| ---- | ----- | -------- | -------- | ------------- | --------- | --------- | -------------- |
| 1 | ReLU-DNN | 0.6723 | 0.7334 | 0.6969 | 0.6711 | 0.7343 | 0.6968 |
| 2 | XGB2 | 0.6712 | 0.7313 | 0.7020 | 0.6758 | 0.7373 | 0.7072 |
| 3 | EBM | 0.6696 | 0.7311 | 0.7012 | 0.6734 | 0.7350 | 0.7057 |
| 4 | GAMI-Net | 0.6690 | 0.7281 | 0.6980 | 0.6698 | 0.7295 | 0.6998 |
| 5 | FIGS | 0.6681 | 0.7274 | 0.6926 | 0.6746 | 0.7371 | 0.6997 |
| 6 | GAM | 0.6669 | 0.7224 | 0.6955 | 0.6669 | 0.7229 | 0.6962 |
| 7 | Tree | 0.6654 | 0.7222 | 0.6976 | 0.6709 | 0.7271 | 0.7033 |
| 8 | GLM | 0.6523 | 0.7066 | 0.6922 | 0.6523 | 0.7077 | 0.6939 |

### Top 3 Models with Hyperparameter and Monotonicity Settings

| Rank | Model | Test ACC | Test AUC | Test F1 Score | Train ACC | Train AUC | Train F1 Score |
| ---- | ----- | -------- | -------- | ------------- | --------- | --------- | -------------- |
| 1 | XGB2 | 0.6715 | 0.7306 | 0.6988 | 0.6748 | 0.7345 | 0.7027 |
| 2 | ReLU-DNN | 0.6676 | 0.7263 | 0.7001 | 0.6681 | 0.7266 | 0.7020 |
| 3 | GAMI-Net | 0.6627 | 0.7169 | 0.6886 | 0.6641 | 0.7180 | 0.6909 |

* **Based on the test ACC metric results, out of the 8 models the best performing models are ReLU-DNN, XGB2, EBM, and GAMI-Net. Further analysis will be continued with the following models: ReLU-DNN, XGB2, and GAMI-Net. GAMI-Net was selected over EBM because XGB2 and EBM operate similarly. After applying hyperparameter and monotonicity settings, XGB2 has the best performance of the three models.**

## Explainability and Interpretability

### XGB2

#### Global Explainability

##### Permutation Feature Importance (PFI)

![XGB2 PFI](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20pfi.png)
* **Amount Past Due is the most influencial feature in this model**
* **Utilization, Mortgage, and Balance have a significant impact on the response variable (Status)**

##### Partial Dependency Plot (PDP) and Accumulated Local Effects (ALE)

* **Amount Past Due**
   * Without Monotonicity:
   ![Amount Past Due](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20amount%20past%20due.png)
   * With Monotonicity:
   ![Amount Past Due](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20monotonic%20amount%20past%20due.png)
   * There is a negative relationship between Status approval and Amount Past Due. As Amount Past Due increases, the probability of that individual not defaulting decreases and defaulting increases.
   
* **Balance**
   * Without Monotonicity:
   ![Balance](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20balance.png)
   * With Monotonicity:
   ![Balance](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20monotonic%20balance.png)
   * There is a positive relationship between Status approval and Balance. As Balance increases, the probability of that individual not defaulting increases and defaulting decreases.
   
* **Credit Inquiry**
   * Without Monotonicity:
   ![Credit Inquiry](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20credit%20inquiry.png)
   * With Monotonicity:
   ![Credit Inquiry](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20monotonic%20credit%20inquiry.png)
   * There is a negative relationship between Status approval and Credit Inquiry. As Credit Inquiry increases, the probability of that individual not defaulting decreases and defaulting increases.
   
* **Delinquency Status**
   * Without Monotonicity:
   ![Delinquency Status](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20delinquency%20status.png)
   * With Monotonicity:
   ![Delinquency Status](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20monotonic%20delinquency%20status.png)
   * There is a negative relationship between Status approval and Delinquency Status. As Delinquency Status increases, the probability of that individual not defaulting decreases and defaulting increases.
   
* **Mortgage**
   * Without Monotonicity:
   ![Mortgage](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20mortgage.png)
   * With Monotonicity:
   ![Mortgage](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20monotonic%20mortgage.png)
   * There is a positive relationship between Status approval and Mortgage. As Mortgage increases, the probability of that individual not defaulting increases and defaulting decreases.
   
* **Utilization**
   * Without Monotonicity:
   ![Utilization](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20utilization.png)
   * With Monotonicity:
   ![Utilization](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20monotonic%20utilization.png)
   * There is a negative relationship between Status approval and Utilization. As Utilization increases, the probability of that individual not defaulting decreases and defaulting increases.

#### Local Explainability

##### LIME and SHAP

![Sample 100](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20lime%20%26%20shap.png)
*  **LIME and SHAP agree that Utilization has the stongest influence on Status for sample 100. For sample 100, Utilization has a negative effect on Status. LIME and SHAP have different rankings and effects for the other features.**

#### Global Interpretability

![XGB2 Global Interpretability](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20global%20interpretability.png)

#### Local Interpretability

![XGB2 Local Interpretability](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20local%20interpretability.png)

### GAMI-Net

#### Global Explainability

##### Permutation Feature Importance (PFI)

![GAMI-Net PFI](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20pfi.png)
* **Delinquency Status is the most influencial feature in this model**
* **Utilization, Mortgage, and Balance have a significant impact on the response variable (Status)**

##### Partial Dependency Plot (PDP) and Accumulated Local Effects (ALE)

* **Balance**
   * Without Monotonicity:
   ![Balance](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20balance.png)
   * With Monotonicity:
   ![Balance](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20monotonic%20balance.png)
   * There is a positive relationship between Status approval and Balance. As Balance increases, the probability of that individual not defaulting increases and defaulting decreases.
   
* **Delinquency Status**
   * Without Monotonicity:
   ![Delinquency Status](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20delinquency%20status.png)
   * With Monotonicity:
   ![Delinquency Status](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20monotonic%20delinquency%20status.png)
   * There is a negative relationship between Status approval and Delinquency Status. As Delinquency Status increases, the probability of that individual not defaulting decreases and defaulting increases.
   
* **Mortgage**
   * Without Monotonicity:
   ![Mortgage](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20mortgage.png)
   * With Monotonicity:
   ![Mortgage](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20monotonic%20mortgage.png)
   * There is a positive relationship between Status approval and Mortgage. As Mortgage increases, the probability of that individual not defaulting increases and defaulting decreases.
  
* **Utilization**
   * Without Monotonicity:
   ![Utilization](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20utilization.png)
   * With Monotonicity:
   ![Utilization](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20monotonic%20utilization.png)
   * There is a negative relationship between Status approval and Utilization. As Utilization increases, the probability of that individual not defaulting decreases and defaulting increases.

#### Local Explainability

##### LIME and SHAP

![Sample 100](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20lime%20%26%20shap.png)
* **

#### Global Interpretability

![GAMI-Net Global Interpretability](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20global%20interpretability.png)

#### Local Interpretability

![GAMI-Net Local Interpretability](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20local%20interpretability%2Cpng.png)

### ReLU-DNN

#### Global Explainability

##### Permutation Feature Importance (PFI)

![ReLU-DNN PFI](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20pfi.png)
* **Delinquency Status is the most influencial feature in this model**
* **Utilization, Mortgage, and Balance have a significant impact on the response variable (Status)**

##### Partial Dependency Plot (PDP) and Accumulated Local Effects (ALE)

* **Amount Past Due**
   ![Amount Past Due](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20amount%20past%20due.png)
   * There is a negative relationship between Status approval and Amount Past Due. As Amount Past Due increases, the probability of that individual not defaulting decreases and defaulting increases.

* **Balance**
   ![Balance](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20balance.png)
   * There is a positive relationship between Status approval and Balance. As Balance increases, the probability of that individual not defaulting increases and defaulting decreases.
   
* **Credit Inquiry**
   ![Credit Inquiry](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20credit%20inquiry.png)
   * There is a negative relationship between Status approval and Credit Inquiry. As Credit Inquiry increases, the probability of that individual not defaulting decreases and defaulting increases.
   
* **Delinquency Status**
   ![Delinquency Status](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20delinquency%20status.png)
   * There is a negative relationship between Status approval and Delinquency Status. As Delinquency Status increases, the probability of that individual not defaulting decreases and defaulting increases.

* **Mortgage**
   ![Mortgage](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20mortgage.png)
   * There is a positive relationship between Status approval and Mortgage. As Mortgage increases, the probability of that individual not defaulting increases and defaulting decreases.
   
* **Utilization**
   ![Utilization](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20utilization.png)
   * There is a negative relationship between Status approval and Utilization. As Utilization increases, the probability of that individual not defaulting decreases and defaulting increases.

#### Local Explainability

##### LIME and SHAP

![Sample 100](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20lime%20%26%20shap.png)
* **

#### Global Interpretability

![ReLU-DNN Global Interpretability](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20global%20interpretability.png)

#### Local Interpretability

![ReLU-DNN Local Interpretability](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20local%20interpretability.png)

## Model Robustness

![All Features](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/model%20robustness%20all%20features.png)
![Amount Past Due](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/model%20robustness%20amount%20past%20due.png)
![Balance](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/model%20robustness%20balance.png)
![Credit Inquiry](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/model%20robustness%20credit%20inquiry.png)
![Delinquency Status](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/model%20robustness%20delinquency%20status.png)
![Mortgage](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/model%20robustness%20mortgage.png)
![Utilization](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/model%20robustness%20utilization.png)
* **The model robustness output indicates that out of the three models, GAMI-Net is the most robust model. The GAMI-Net model reacts well to the imputation of noise into the data. The XGB2 model has the highest accuracy associated with 0 pertubation (noise), but drops rapidly once noise is inputed into the data. After analyzing specific features, the XGB2 model responds the best to noise imputation for all features except Amount Past Due and Balance.**

## Model Fairness

### XGB2 Adverse Impact Ratio (AIR)

| Protected Groups | Feature Configuration | AIR | ACC |
| ---------------- | --------------------- | --- | --- |
| Race 0 | Mortgage | 0.956147 | 0.662930 |
| Gender 0 | Mortgage | 0.928242 | 0.662930 |

![Binned Mortgage for Race and Gender](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20fairness.png)

### GAMI-Net Adverse Impact Ratio (AIR)

| Protected Groups | Feature Configuration | AIR | ACC |
| ---------------- | --------------------- | --- | --- |
| Race 0 | Mortgage | 0.913163 | 0.657870 |
| Gender 0 | Balance | 0.903219 | 0.661800 |

![Binned Mortgage for Race](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20fairness%20race.png)

![Binned Balance for Gender](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20fairness%20gender.png)

### ReLU-DNN Adverse Impact Ratio (AIR)

| Protected Groups | Feature Configuration | AIR | ACC |
| ---------------- | --------------------- | --- | --- |
| Race 0 | Mortgage | 0.908938 | 0.664070 |
| Gender 0 | Mortgage | 0.979822 | 0.664070 |

![Binned Mortgage for Race and Gender](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20fairness.png)

## Model Comparison to Black-Box Modeling

## Potential Next Steps
* **Apply this end-to-end model process on a real dataset**
   * Potentially launch this model with a small bank or credit union
   * Adjust additional hyperparameter settings

## Risk Considerations and Ethical Concerns
* **PiML toolbox is a risk aware machine learning package**
* **PiML was used to mitigate risks and build models that address and find the balance between the following metrics**:
   * Accuracy
   * Bias/Fairness
   * Reliability
   * Robustness
   * Resilience
   * Transparency
* **Potential negative impacts of using this model**:
   * The final model produced an accuracy of 
* **Potential uncertainties relating to the impacts of using this model**:
   * The model will need to be monitored into the future for its fairness, performance, and robustness to ensure it is still operating efficiently and effectively.
   * The model contains personal data that should be anonymized to protect the privacy of customers. There should be security in place to prevent hackers and malicious software from gaining unauthorized access and altering the code for malicious intent.
