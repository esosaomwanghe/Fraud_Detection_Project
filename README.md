# Fraud_Detection_Project
Machine learning-based fraud detection system for digital money transfers using transaction, behavioral, device, and risk analytics. The project focuses on real-time fraud prediction, false-positive reduction, and scalable fraud monitoring for fintech platforms.


## Business Challenges

### 1. Increasing Sophistication of Fraud Attacks
Digital money transfer platforms such as NovaPay face increasingly sophisticated fraud schemes, including identity theft, unauthorized transactions, account takeovers, and suspicious cross-border fund transfers. Traditional rules-based fraud detection systems are no longer sufficient to identify evolving fraud patterns in real time.

### 2. High Financial and Operational Losses
Fraudulent transactions can result in substantial financial losses through chargebacks, reimbursement claims, transaction reversals, and regulatory penalties. As transaction volumes increase globally, even a small percentage of fraud can significantly impact operational costs and profitability.

### 3. Poor Scalability of Static Fraud Detection Systems
The existing static rules-based fraud detection approach lacks scalability and adaptability. Maintaining manually defined fraud rules becomes increasingly difficult as:
- Customer bases expand internationally
- Transaction volumes grow rapidly
- Fraud tactics evolve continuously

This limits the organization’s ability to respond quickly to emerging fraud threats.

### 4. High False Positive Rates
A major challenge in fraud detection is minimizing false positives, where legitimate customer transactions are incorrectly flagged as fraudulent. High false positive rates can:
- Interrupt customer transactions
- Reduce customer satisfaction
- Increase customer churn
- Generate unnecessary manual investigations

The business requires a more intelligent system capable of distinguishing legitimate transactions from fraudulent behavior more accurately.

### 5. Need for Real-Time Fraud Detection
Digital money transfer systems operate in real-time environments where transactions must be approved or rejected instantly. Fraud detection models must therefore:
- Process transactions with low latency
- Detect suspicious activity immediately
- Support automated decision-making
- Integrate seamlessly into transaction systems

### 6. Regulatory Compliance and Explainability
Financial institutions must comply with anti-money laundering (AML), know-your-customer (KYC), and fraud prevention regulations. The fraud detection system must provide:
- Explainable fraud predictions
- Transparent decision-making
- Audit-ready outputs
- Regulatory compliance support

### 7. Customer Trust and Brand Reputation
Frequent fraud incidents or blocked legitimate transactions can damage customer confidence and negatively affect the organization’s reputation. Maintaining a secure, reliable, and frictionless transaction experience is critical for customer retention and long-term business growth.

---

# Target Variables

## Primary Target Variable

| Variable | Description |
|---|---|
| `is_fraud` | Indicates whether a transaction is fraudulent (`1`) or legitimate (`0`) |

The `is_fraud` variable serves as the primary supervised learning target for the fraud detection model.

---

# Additional Fraud Intelligence Target Variables

The following high-impact variables can also be used as analytical or derived target variables to support advanced fraud detection, behavioral analysis, and operational risk monitoring.

| Variable | Description | Business Relevance |
|---|---|---|
| `risk_score_internal` | Internal fraud risk score assigned to transactions | Supports real-time fraud risk assessment |
| `ip_risk_score` | Risk score associated with the transaction IP address | Helps identify suspicious network activity |
| `device_trust_score` | Trustworthiness score of the customer device | Detects compromised or unknown devices |
| `txn_velocity_1h` | Transaction frequency within 1 hour | Identifies rapid suspicious activity |
| `txn_velocity_24h` | Transaction frequency within 24 hours | Detects abnormal transaction patterns |
| `location_mismatch` | Indicates mismatch between customer location and transaction location | Helps detect account takeover and location spoofing |
| `new_device` | Indicates whether a new device was used | Detects suspicious login/device changes |
| `amount_usd` | Transaction amount in USD | High-value transactions often carry higher fraud risk |
| `corridor_risk` | Risk level associated with transaction corridor or country pair | Detects high-risk geographic transfer routes |
| `chargeback_history_count` | Number of previous customer chargebacks | Strong indicator of repeated fraudulent behavior |

---

# Project Objective

The objective of this project is to develop a machine learning-based fraud detection system capable of:
- Detecting fraudulent transactions accurately
- Reducing false positives
- Supporting real-time fraud monitoring
- Scaling with increasing transaction volumes
- Adapting to evolving fraud tactics
- Providing explainable and regulatory-compliant fraud predictions

The solution will replace the current static rules-based approach with a scalable, intelligent, and adaptive fraud detection framework.




# Models Used for the Fraud Detection Project

The following machine learning models were implemented and evaluated for fraud detection:

1. **Logistic Regression (Baseline Model)**

   - Used as the baseline classification model.
   - Provides interpretable linear classification results.

2. **XGBoost Classifier**

   - Gradient boosting algorithm capable of capturing complex fraud patterns.
   - Handles nonlinear feature interactions effectively.

3. **Hyperparameter-Tuned XGBoost**

   - Optimized version of XGBoost using hyperparameter tuning techniques.
   - Improved fraud recall performance.

4. **SHAP (SHapley Additive Explanations)**

   - Implemented for model explainability.
   - Used to explain global and individual fraud predictions.

---

# Results and Discussion of Findings

## Logistic Regression Performance

| Metric | Score |
|---|---|
| Accuracy | 98.61% |
| Precision | 0.9895 |
| Recall | 0.8670 |
| F1-Score | 0.9242 |
| ROC-AUC | 0.9601 |

### Discussion

The Logistic Regression model achieved the best overall performance among all evaluated models. The model maintained extremely high accuracy and precision while achieving strong fraud recall capability. The ROC-AUC score of 0.960 demonstrates excellent ability to distinguish between fraudulent and legitimate transactions.

The strong performance suggests that the engineered fraud indicators created highly separable fraud patterns, allowing the simpler linear model to generalize effectively while minimizing false positives.

---

## XGBoost Performance

| Metric | Score |
|---|---|
| Accuracy | 98.43% |
| Precision | 0.9692 |
| Recall | 0.8670 |
| F1-Score | 0.9153 |
| ROC-AUC | 0.9472 |

### Discussion

The XGBoost model achieved performance very close to Logistic Regression while offering improved capability for learning nonlinear fraud patterns and feature interactions. Although its ROC-AUC score was slightly lower than Logistic Regression, the model still demonstrated strong fraud classification capability.

Feature importance analysis revealed that transaction velocity, internal risk scores, IP risk indicators, and account age were the most influential fraud predictors.

---

## Tuned XGBoost Performance

| Metric | Score |
|---|---|
| Accuracy | 96.86% |
| Precision | 0.8136 |
| Recall | 0.8807 |
| F1-Score | 0.8458 |
| ROC-AUC | 0.9533 |

### Discussion

The tuned XGBoost model achieved the highest fraud recall score among all models, meaning it detected the largest proportion of fraudulent transactions. This makes it valuable in high-risk fraud monitoring environments where missing fraudulent activity is more costly than generating false alerts.

However, the improvement in recall came at the expense of lower precision and overall accuracy, leading to a higher number of false positives. This demonstrates the typical trade-off between fraud sensitivity and operational efficiency.

---

# Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9861 | 0.9895 | 0.8670 | 0.9242 | 0.9601 |
| XGBoost | 0.9843 | 0.9692 | 0.8670 | 0.9153 | 0.9472 |
| Tuned XGBoost | 0.9686 | 0.8136 | 0.8807 | 0.8458 | 0.9533 |

---

# Model Interpretation

Although XGBoost is generally considered a more advanced machine learning algorithm, Logistic Regression slightly outperformed XGBoost in this project. This indicates that the engineered fraud features, including transaction velocity, risk scores, behavioral indicators, and location mismatch variables, created strong linear separability within the dataset.

Consequently, the simpler Logistic Regression model generalized exceptionally well while maintaining low false positive rates. The findings demonstrate that simpler models can outperform more complex ensemble methods when high-quality predictive features are available.

The tuned XGBoost model slightly improved fraud recall, meaning it detected more fraudulent transactions. However, this improvement came with reduced precision and increased false positive alerts. This highlights the operational trade-off between maximizing fraud detection sensitivity and minimizing disruptions to legitimate customer transactions.

---

# SHAP Explainability Findings

SHAP (SHapley Additive exPlanations) was implemented to improve model interpretability and explain individual fraud predictions.

The SHAP analysis revealed the following major fraud drivers:

1. **Transaction Velocity (1-hour and 24-hour windows)**  
   Rapid consecutive transactions strongly increased fraud probability.

2. **Internal Risk Score**  
   Higher internal fraud risk scores significantly contributed to fraud classification.

3. **IP Risk Score**  
   Transactions originating from risky IP addresses showed increased fraud likelihood.

4. **Account Age**  
   Newer accounts were more likely to be associated with fraudulent activity.

5. **Location Mismatch**  
   Transactions where the IP country differed from the registered home country contributed positively to fraud risk.

SHAP waterfall plots also provided transaction-level explanations by showing how individual features increased or decreased fraud prediction scores for specific transactions.

---

# Key Insights

1. **Transaction velocity is the strongest fraud indicator**

   Fraudulent users frequently perform rapid consecutive transactions within short periods.

2. **High IP risk scores strongly correlate with fraud**

   Suspicious IP addresses significantly increase fraud probability.

3. **Internal risk scoring systems are highly predictive**

   Internal fraud risk scores contributed heavily to model decisions.

4. **Newer accounts show higher fraud tendencies**

   Accounts with lower account age were more likely to perform fraudulent transactions.

5. **Location mismatch contributes to fraud risk**

   Transactions where IP country differed from home country showed elevated fraud likelihood.

6. **SHAP explainability improved model transparency**

   SHAP plots clearly explained why specific transactions were classified as fraudulent or legitimate.

7. **Simpler models can outperform complex models**

   Logistic Regression achieved the best overall performance despite being less complex than XGBoost.

---

# Business Recommendations Based on the Findings

## 1. Implement Real-Time Transaction Velocity Monitoring

Financial institutions should continuously monitor rapid transaction activity within short time windows to identify suspicious behavior early.

## 2. Strengthen IP Risk Intelligence Systems

Integrate external IP reputation databases and geolocation intelligence tools to improve fraud prevention capabilities.

## 3. Increase Monitoring for New Accounts

Newly created customer accounts should undergo stricter fraud screening and enhanced verification procedures.

## 4. Deploy Explainable AI Dashboards

SHAP-based explainability dashboards should be integrated into fraud monitoring systems to help analysts understand model decisions.

## 5. Optimize Fraud Detection Thresholds

Organizations should adjust fraud thresholds depending on operational priorities:

- Higher precision reduces false alarms
- Higher recall improves fraud detection coverage

## 6. Prioritize Behavioral Risk Analytics

Behavioral indicators such as transaction frequency, account activity patterns, and location consistency should be heavily prioritized in fraud detection systems.

## 7. Continuously Retrain Fraud Models

Fraud behavior evolves over time. Organizations should continuously retrain and monitor fraud detection models to maintain prediction accuracy and adapt to emerging fraud strategies.

---

# Conclusion

This project successfully developed a machine learning-based fraud detection system capable of accurately identifying fraudulent digital money transfer transactions.

Logistic Regression achieved the best overall performance, demonstrating that carefully engineered behavioral and risk-based features created strong fraud separability within the dataset. XGBoost models provided additional nonlinear learning capability, while SHAP explainability improved model transparency by identifying the most influential fraud drivers.

The findings demonstrated that transaction velocity, internal risk scores, IP risk indicators, account age, and location mismatch are major contributors to fraud prediction. Overall, the project provides a scalable, interpretable, and production-ready fraud detection framework suitable for real-time financial risk monitoring and fraud prevention systems.