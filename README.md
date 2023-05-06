# Wells Fargo Credit Modeling Model Card

## Basic Information

* **Organization Developing Model**: GWU Wells Fargo Credit Modeling Team
* **Model Date**: May, 2023
* **Model Version**: 1.0
* **License**: Apache 2.0
* **Model Implementation Code**: [Wells_Fargo.ipynb](https://github.com/tariaherbert/Wells-Fargo/blob/main/Wells_Fargo.ipynb) 

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

* **Source of training data**: Wells Fargo
* **How training data was randomly split into training and testing data**: 80% training and 20% testing
* **Number of rows in training and testing data**:
  * Training rows: 80,000
  * Testing rows: 20,000

## Model Details
* **Columns used as inputs in the final model**: 'Mortgage', 'Balance', 'Amount Past Due', 'Delinquency Status', 'Credit Inquiry', 'Utilization'
* **Column used as target in the final model**: 'Status'
* **Type of final model**: GAMI-Net
* **Software used to implement the models**: Numpy, PiML, Python, XGBoost
* **Version of the modeling software**: 
   * Numpy version: 1.23.5
   * PiML version: 4.3
   * Python version: 3.9.16
   * XGBoost version: 1.7.2
* **Hyperparameters or other settings of the final model**:
   * N_interactions: default 10
   * Batch_size: default 1000
   * Subnet1_size: Set to 5 (default 20)
   * Subnet2_size: Set to 5 (default 20)
   * Max_epochs: default 1000
   * Learning_rates: default 1e-3, 1e-3, 1e-4
   * Random_state: default 0
   * Monotonic Settings: monotonic increasing ('Balance', 'Mortgage'), monotonic decreasing ('Amount Past Due',  'Credit Inquiry', 'Delinquency Status', 'Utilization')

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

## Methodology

### Classification and Regression Decision Trees (CART)

* **CART models are developed by selecting predictive variables and evaluating the different split points of each predictor until the ideal tree is produced.**
* **Formula: f(x) = 1 - Σ (Pi)2**
* **Advantages**:
   * Interpretable for Relatively Moderate-Sized Trees
   * Flexible (classification and regression)
   * Relatively Robust Against Noise
* **Disadvantages**:
   * Unstable
   * Difficult to Understand Individual Feature Contribution
   * Interpretability Declines for Larger Trees
   * Prone to Overfitting
   * High Variance

### Explainable Boosting Machine (EBM)

* **EBM models are similar to GAM models, however unlike GAM models, EBM models can automatically detect and include pairwise interaction terms.**
* **Formula: g(E[y]) = b0 + ∑ fj(xj) + ∑ fij(xi,xj)**
* **Advantages**:
   * Better predictive power
   * One of the fastest models to execute at prediction time
   * Light memory usage
   * Fast computation
   * Nice visualization
   * Good support from Microsoft Research
* **Disadvantages**:
   * Runs slower than other models
   * Non-smooth and jumpy shape functions
   * Lacking monotonicity constraint
   * Lacking pruning for main effects

### Fast Interpretable Greedy-Tree Sums (FIGS)

* **FIGS models are developed by adding predictor variables one by one while considering the split for an ensemble of trees.**
* **Formula: f(x) = Σ fk(Pk(x)) + f0**
* **Advantages**:
   * Interpretable for Relatively Small-Sized Trees
   * Flexible (used for both classification and regression)
   * Relatively Robust Against Noise
   * More Stable than CART models
   * Decouples Feature Interactions
* **Disadvantages**:
   * Less Interpretable than CART Models (Difficult to Follow Multiple Separate Trees)
   * Prone to Overfitting
   * High Variance

### Generalized Additive Model (GAM)

* **GAM is a linear model with a key difference when compared to Generalised Linear Models such as Linear Regression. A GAM is allowed to learn non-linear features. The sum of many splines forms a GAM. The result is a highly flexible model which still has some of the explainability of a linear regression.**
* **Formula: y = g(μ) = b0 + f(x1) + f(x2)... + f(xp)**
* **Advantages**:
   * Very flexible in choosing non-linear models
   * Because of the additivity we can still interpret the contribution of each predictor while considering the other predictors fixed.
   * GAMs may outperform linear models in terms of prediction.

### Generalized Additive Model with Structured Interactions (GAMI-Net)

* **GAMI-Net models are similar to GAM models as they both**:
   * captures non linear relationships, non monotonic patterns using splines
* **and**
   * lack interpretability if the smooth functions are complex. GAMI-Net is different than GAM because it models the interactions between variables using a neural networks smoothing approach, which allows the model to capture complex non-linear interactions between variables.
* **Formula: g(E(y|x)) = μ + ∑ hj(xj) + ∑ f jk (xj,xk)**
* **Advantages**:
   * Flexible, it captures complex interactions which improves accuracy
   * Handles linear and non linear relationships between variables
* **Disadvantages**:
   * Computationally intensive
   * It can be a non interoperable model depending on features selection due to its neural network approach

### Generalized Linear Model (GLM)

* **GLM is a flexible generalization of ordinary linear regression. The GLM model generalizes linear regression by allowing the linear model to be related to the response variable via a link function and by allowing the magnitude of the variance of each measurement to be a function of its predicted value.**
* **Formula: y =  β0 +β1x1i + ... + βpxpi**
* **Advantages**:
   * Very easy to understand and explain
* **Disadvantages**:
   * Tends to focus only on linear relationships, Unable to detect nonlinearity directly 
   * Lower performance in comparison to other models

### Deep ReLU Networks using Aletheia Unwrapper and Sparsification (ReLU-DNN)

* **The ReLU-DNN model is a type of deep neural network that uses the Rectified Linear Unit (ReLU) activation function in its hidden layers. The ReLU function allows the network to learn complex, non-linear relationships between the input data and the output labels. During training, the weights of the network are updated using an optimization algorithm to minimize a loss function that measures the difference between the predicted outputs and the true labels.**
* **Formula: f(x) = max(0,x)**
* **Advantages**:
   * Effective at learning non-linear relationships between inputs and outputs
   * Suitable for large-scale, complex datasets
   * Relatively fast convergence during training
   * Avoids the vanishing gradient problem that can occur in deep neural networks when using other activation functions
   * Has achieved state-of-the-art performance on image classification, speech recognition, and natural language processing
* **Disadvantages**:
   * May require large amounts of training data and computational resources
   * Can be sensitive to the choice of hyperparameters, such as the learning rate and regularization strength
   * Can be difficult to interpret and understand the learned representations

### Extreme Gradient Boosted Trees of Depth 2 (XGB2)

* **XGB2 refers to a specific variant of the Extreme Gradient Boosted (XGB) Forests model. XGB2 limits the depth of each tree to 2 splits, which allows for pairwise interactions between features.**
* **Formula: f(x) = ∑ fk(xi)**
* **Advantages**:
   * Handles non-linear relationships
   * Distinguishes interactions between variables better than linear models
* **Disadvantages**:
   * Prone to overfitting

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

### Global Explainability

#### Permutation Feature Importance (PFI)

* **Permutation Feature Importance (PFI): Computes the change in prediction performance as the measure of feature importance**
   * Breaks the relationship between the feature and the target, thus the drop in the model score is indicative of how much the model depends on the feature

##### GAMI-Net

![GAMI-Net PFI](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20pfi.png)
* **Delinquency Status is the most influencial feature in this model**
* **Utilization, Mortgage, and Balance have a significant impact on the response variable (Status)**

##### ReLU-DNN

![ReLU-DNN PFI](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20pfi.png)
* **Delinquency Status is the most influencial feature in this model**
* **Utilization, Mortgage, and Balance have a significant impact on the response variable (Status)**

##### XGB2

![XGB2 PFI](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20pfi.png)
* **Amount Past Due is the most influencial feature in this model**
* **Utilization, Mortgage, and Balance have a significant impact on the response variable (Status)**

#### Partial Dependency Plot (PDP) and Accumulated Local Effects (ALE)

* **Partial Dependence Plot (PDP): used to understand how the prediction varies as a function of variables of interest, by averaging over other variables**
   * Not recommended if the features are correlated
* **Accumulated Local Effects (ALE): describes how features affect a model prediction**
   * Shares the same goal as PDP (Partial Dependence Plot)
   * Overcomes the features correlation problem by averaging and accumulating the difference in predictions across the conditional distribution, limiting the effects of specific features

##### GAMI-Net

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

##### ReLU-DNN

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

##### XGB2

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

### Local Explainability

#### LIME and SHAP

* **LIME: Explains creating a surrogate (smaller, simpler)  model. Then approximates predictions**
   * The "effect" in LIME refers to the impact of each feature on the predicted outcome for the selected instance. Positive coefficients indicate that the feature has a positive effect on the prediction, while negative coefficients indicate a negative effect.
   * The "weight" in LIME refers to the importance of each feature in the overall explanation.
* **SHAP:** 
   * F(x) refers to the prediction of the machine learning model for a given data point x, while E[F(x)] refers to the expected prediction of the model over a background dataset. 
   * If F(x) and E[F(x)] are close to each other, it suggests that the model is making predictions that are consistent with the overall behavior of the dataset. 
   * If there is a large difference between F(x) and E[F(x)], it suggests that the model makes predictions that are significantly different from the overall behavior of the dataset, which may indicate bias or overfitting.
   * Positive values indicate that the feature contributes positively to the prediction, while negative values indicate that the feature contributes negatively. The size of the SHAP value indicates the magnitude of the feature's influence.

##### GAMI-Net

![GAMI-Net Sample 100](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20lime%20%26%20shap.png)
* **LIME and SHAP disagree on the most influencial feature on Status for Sample 100. LIME suggests that Utilization is the most influential feature with a negative effect on Status, while SHAP suggests that Delinquency Status is the most influencial with a positive effect on Status.**

##### ReLU-DNN

![ReLU-DNN Sample 100](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20lime%20%26%20shap.png)
* **LIME and SHAP agree on all the rankings and effects of the features. For Sample 100, Delinquency Status is indicated to be the most influencial feature on Status.**

##### XGB2

![XGB2 Sample 100](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20lime%20%26%20shap.png)
*  **LIME and SHAP agree that Utilization has the stongest influence on Status for sample 100. For sample 100, Utilization has a negative effect on Status. LIME and SHAP disagree on the rankings and effects of the other features on Status.**

### Global Interpretability

* **Refers to the ability to understand how a machine learning model makes predictions across the entire dataset or population.**
* **Involves techniques that help to identify the most important features or patterns in the data that contribute to the model's predictions.**
* **Technique(s): feature importance values.**
* **Helps identify biases or issues in the model and provides insights for improving the model's performance.**

#### GAMI-Net

![GAMI-Net Global Interpretability](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20global%20interpretability.png)
* **The most influencial features when predicting Status are Delinquency Status, Utilization, Mortgage, and Balance.**

#### ReLU-DNN

![ReLU-DNN Global Interpretability](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20global%20interpretability.png)
* **The most influencial features when predicting Status are Mortgage, Delinquency Status, and Balance.**

#### XGB2

![XGB2 Global Interpretability](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20global%20interpretability.png)
* **The most influencial features when predicting Status are Amount Past Due, Utilization, Mortgage, and Balance.**

### Local Interpretability

* **Refers to the ability to understand how a machine learning model makes predictions for individual inputs or instances.**
* **Involves techniques that help identify the most important features or patterns in the data that contribute to the model's prediction for a specific input.**
* **Technique(s): local feature importance, local effect importance, and surrogate models.**
* **Helps build trust in the model by providing explanations for individual predictions and provides insights for improving the model's performance at the local level.**

#### GAMI-Net

![GAMI-Net Local Interpretability](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20local%20interpretability%2Cpng.png)

#### ReLU-DNN

![ReLU-DNN Local Interpretability](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20local%20interpretability.png)

#### XGB2

![XGB2 Local Interpretability](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20local%20interpretability.png)

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

* **Model fairness is vital to selecting the right model by ethical and regulatory standards. A model should be independent of an individual's race and gender. Discrimination is faced by groups as a whole or even as individuals who do not receive similar treatment. We measured model fairness using the Adverse Impact Ratio (AIR). The general threshold for AIR is 0.8, but banks prefer to use a threshold of 0.9. There are multiple methods in PiML to adjust for AIR, such as segmenting, thresholding, binning and feature removal. In this case, the team has opted to use the binning method.**

### GAMI-Net Adverse Impact Ratio (AIR)

| Protected Groups | Feature Configuration | Original AIR | Original ACC| Binned AIR | Binned ACC |
| ---------------- | --------------------- | ------------ | ----------- | ---------- | ---------- |
| Gender 0 | Mortgage | 0.871185 | 0.663810 | 0.918726 | 0.657870 |
| Race 0 | Mortgage | 0.646503 | 0.663810 | 0.913163 | 0.657870 |

![Binned Mortgage for Race and Gender](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20fairness.png)

### ReLU-DNN Adverse Impact Ratio (AIR)

| Protected Groups | Feature Configuration | Original AIR | Original ACC| Binned AIR | Binned ACC |
| ---------------- | --------------------- | ------------ | ----------- | ---------- | ---------- |
| Gender 0 | Mortgage | 0.915365 | 0.668010 | 0.979822 | 0.664070 |
| Race 0 | Mortgage | 0.683560 | 0.668010 | 0.908938 | 0.664070 |

![Binned Mortgage for Race and Gender](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20fairness.png)

### XGB2 Adverse Impact Ratio (AIR)

| Protected Groups | Feature Configuration | Original AIR | Original ACC| Binned AIR | Binned ACC |
| ---------------- | --------------------- | ------------ | ----------- | ---------- | ---------- |
| Gender 0 | Mortgage | 0.821672 | 0.674150 | 0.928242 | 0.662930 |
| Race 0 | Mortgage | 0.603988 | 0.674150 | 0.956147 | 0.662930 |

![Binned Mortgage for Race and Gender](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20fairness.png)

## New Dataset Generation

### Model Performance

| Rank | Model | Test ACC | Test AUC | Test F1 Score | Train ACC | Train AUC | Train F1 Score |
| ---- | ----- | -------- | -------- | ------------- | --------- | --------- | -------------- |
| 1 | XGB2 | 0.6640 | 0.7125 | 0.7062 | 0.6649 | 0.7151 | 0.7085 |
| 2 | ReLU-DNN | 0.6608 | 0.7023 | 0.7034 | 0.6615 | 0.7013 | 0.7056 |
| 3 | GAMI-Net | 0.6587 | 0.7044 | 0.6966 | 0.6599 | 0.7040 | 0.6987 |

### Model Robustness

![New Model Robustness](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/new%20model%20robustness.png)

### Model Fairness

#### GAMI-Net Adverse Impact Ratio (AIR)

| Protected Groups | AIR |
| ---------------- | --- |
| Gender 0 | 0.931581 |
| Race 0 | 0.911592 |

![GAMI-Net new fairness](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/GAMI-Net%20new%20fairness.png)

#### ReLU-DNN Adverse Impact Ratio (AIR)

| Protected Groups | AIR |
| ---------------- | --- |
| Gender 0 | 0.990895 |
| Race 0 | 0.954666 |

![ReLU-DNN new fairness](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/ReLU-DNN%20new%20fairness.png)

#### XGB2 Adverse Impact Ratio (AIR)

| Protected Groups | AIR |
| ---------------- | --- |
| Gender 0 | 0.916539 |
| Race 0 | 0.933909 |

![XGB2 new fairness](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/XGB2%20new%20fairness.png)

### Model Comparison to Black-Box Modeling

* **Black box models are structured in a manner such that it takes in the input and gives an output without showing the user how the output was arrived at. In other words, it is not transparent, explainable or interpretable. In the case of credit modeling if it were to accept or reject an individual the user of the model would not be able to explain the reason for accepting or rejecting any particular individual. Similarly, it is possible to test the fairness and robustness of these models, but it would not be easy for the user to understand why the model was unfair or not robust. These models are highly complex hence they are difficult to debug and edit but also due to the complexity there are higher chances of it being manipulated without leaving a trace.** 

![LGMB Robustness](https://github.com/tariaherbert/Wells-Fargo/blob/main/graphs/LGMB%20robustness.png)
* **This model robustness output indicates that the LGMB classifier is slightly less robust than the XGB2 model and has the second highest accuracy when there is zero pertubation. There is no logical reason to use the black box model since it is not transparent and highly complex and underperforms comapared to the interpretable XGB2 model.** 

## Ethical Concerns
* **Potential negative impacts of using this model**:
   * The final model has an accuracy of 65.87%, which means there is a 34.13% chance that the model makes inaccurate decisions. This can become costly for the company.
* **Potential uncertainties relating to the impacts of using this model**:
   * The model will need to be monitored into the future for its fairness, performance, and robustness to ensure it is still operating efficiently and effectively.
   * The model contains personal data that should be anonymized to protect the privacy of customers. There should be security in place to prevent hackers and malicious software from gaining unauthorized access and altering the code for malicious intent.

## Risk Considerations
* **PiML toolbox is a risk aware machine learning package**
* **PiML was used to mitigate risks and build models that address and find the balance between the following metrics**:
   * Accuracy
   * Bias/Fairness
   * Reliability
   * Robustness
   * Resilience
   * Transparency

## Potential Next Steps
* **Apply this end-to-end model process on a real dataset**
   * Potentially launch this model with a small bank or credit union
* **Adjust additional hyperparameter settings**
* **Run a cost benefit analysis**

## Author Contributions
* **TH served as the primary contributor for the model card. PS served as the primary contributor for the PiML colaboratory notebook. PS & BA contributed equally when analyzing GAMI-Net. JC & LZ contributed equally when analyzing ReLU-DNN. TH & RJ contributed equally when analyzing XGB2. PS, TH, RJ, & BA contributed equally when analyzing model fairness. PS & RJ contributed when testing for bias and generating new dataset. All members contributed equally when analyzing model performance and robustness. 
