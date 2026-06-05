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

| Metric    | Score  |
| ---------- | ------- |
| Accuracy  | 98.38% |
| Precision | 1.00   |
| Recall    | 0.82   |
| F1-Score  | 0.90   |
| ROC-AUC   | 0.95   |

### Discussion

The Logistic Regression model achieved very high accuracy and precision, indicating excellent ability to correctly identify legitimate transactions while minimizing false positives. The model maintained strong fraud detection capability with a recall score of 82%.

---

## XGBoost Performance

| Metric    | Score  |
| ---------- | ------- |
| Accuracy  | 98.34% |
| Precision | 0.99   |
| Recall    | 0.82   |
| F1-Score  | 0.90   |
| ROC-AUC   | 0.93   |

### Discussion

The XGBoost model performed similarly to logistic regression while providing better capability for capturing nonlinear fraud patterns and complex feature relationships. Feature importance analysis showed that transaction velocity and risk-related variables were the strongest fraud indicators.

---

## Tuned XGBoost Performance

| Metric    | Score  |
| ---------- | ------- |
| Accuracy  | 96.59% |
| Precision | 0.80   |
| Recall    | 0.83   |
| F1-Score  | 0.81   |
| ROC-AUC   | 0.93   |

### Discussion

The tuned XGBoost model slightly improved fraud recall, meaning it detected more fraudulent transactions. However, this improvement came at the cost of lower precision and overall accuracy, resulting in more false positive alerts.

---

# Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9838 | 1.00 | 0.82 | 0.90 | 0.946 |
| XGBoost | 0.9834 | 0.99 | 0.82 | 0.898 | 0.931 |
| Tuned XGBoost | 0.9659 | 0.80 | 0.83 | 0.813 | 0.935 |


---

# Model Interpretation

Although XGBoost is generally considered a more advanced machine learning algorithm, Logistic Regression slightly outperformed XGBoost in this project. This suggests that the engineered fraud features, including transaction velocity, risk scores, and behavioral indicators, created strong linear separation between fraudulent and legitimate transactions.

Consequently, the simpler Logistic Regression model was able to generalize effectively while minimizing false positives. The findings demonstrate that simpler models can outperform more complex ensemble methods when the dataset contains highly informative and well-structured predictive features.

The tuned XGBoost model slightly improved fraud recall, meaning it detected more fraudulent transactions, but this improvement came at the expense of reduced precision and increased false positives. This highlights the trade-off between maximizing fraud detection sensitivity and minimizing disruption to legitimate customer transactions.

---

# Key Insights

1. **Transaction velocity is the strongest fraud indicator**

   - Fraudulent users frequently perform rapid consecutive transactions within short periods.

2. **High IP risk scores strongly correlate with fraud**

   - Risky or suspicious IP addresses significantly increase fraud probability.

3. **Internal risk scoring systems are highly predictive**

   - The internal fraud risk score contributed heavily to fraud classification decisions.

4. **Newer accounts show higher fraud tendencies**

   - Accounts with lower account age were more likely to perform fraudulent transactions.

5. **Location mismatch contributes to fraud risk**

   - Transactions where IP country differs from home country showed increased fraud likelihood.

6. **SHAP explainability improved model transparency**

   - SHAP plots clearly explained why specific transactions were classified as fraudulent or legitimate.

---

# Business Recommendations Based on the Findings

## 1. Implement Real-Time Transaction Velocity Monitoring

Financial institutions should monitor rapid transaction activity within 1-hour and 24-hour windows to detect suspicious behavior early.

## 2. Strengthen IP Risk Intelligence Systems

Integrate external IP reputation databases and geolocation intelligence to improve fraud prevention.

## 3. Increase Monitoring for New Accounts

Newly created customer accounts should undergo stricter fraud screening and enhanced verification procedures.

## 4. Deploy Explainable AI Dashboards

SHAP-based explainability dashboards should be integrated into fraud monitoring systems to help analysts understand prediction decisions.

## 5. Optimize Fraud Detection Thresholds

Organizations should balance precision and recall depending on operational goals:

- Higher precision reduces false alarms
- Higher recall detects more fraud cases

## 6. Incorporate Behavioral Risk Analytics

Behavioral indicators such as transaction frequency, account activity patterns, and location consistency should be prioritized in fraud monitoring systems.

---

# Conclusion

This project successfully developed a machine learning-based fraud detection system capable of accurately identifying fraudulent digital money transfer transactions. Logistic Regression provided strong baseline performance, while XGBoost captured more complex fraud behavior patterns. SHAP explainability further enhanced model transparency by identifying the key factors driving fraud predictions. The findings demonstrate that transaction velocity, risk scores, account age, and IP-related indicators are major fraud drivers. Overall, the project provides a scalable and explainable fraud detection framework that can support real-time financial risk management and fraud prevention systems.