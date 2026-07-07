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
   - Improved precision, F1-score, and ROC-AUC performance.

4. **SHAP (SHapley Additive Explanations)**

   - Implemented for model explainability.
   - Used to explain global and individual fraud predictions.

---

# Results and Discussion of Findings

## Logistic Regression Performance

| Metric | Score |
|---|---|
| Accuracy | 95.33% |
| Precision | 0.7050 |
| Recall | 0.8991 |
| F1-Score | 0.7903 |
| ROC-AUC | 0.9638 |

### Discussion

The Logistic Regression model achieved strong fraud detection capability with the highest recall score among the evaluated models. This indicates that the model was highly effective at identifying fraudulent transactions. However, the lower precision score suggests that the model generated a higher number of false positive alerts compared to the XGBoost-based models.

Despite being a simpler linear model, Logistic Regression demonstrated excellent separability between fraudulent and legitimate transactions due to the strength of the engineered behavioral and risk-based fraud features.

---

## XGBoost Performance

| Metric | Score |
|---|---|
| Accuracy | 98.38% |
| Precision | 0.9643 |
| Recall | 0.8670 |
| F1-Score | 0.9130 |
| ROC-AUC | 0.9602 |

### Discussion

The XGBoost model achieved excellent classification performance with very high precision and accuracy. The model effectively captured nonlinear relationships and interactions among transaction behavior, device intelligence, and risk indicators.

Compared to Logistic Regression, XGBoost significantly reduced false positive alerts while maintaining strong fraud detection capability. This makes the model operationally efficient for large-scale fraud monitoring systems.

---

## Tuned XGBoost Performance

| Metric | Score |
|---|---|
| Accuracy | 98.70% |
| Precision | 1.0000 |
| Recall | 0.8670 |
| F1-Score | 0.9287 |
| ROC-AUC | 0.9698 |

### Discussion

The Tuned XGBoost model achieved the best overall performance among all evaluated models. It produced the highest accuracy, precision, F1-score, and ROC-AUC score while maintaining strong fraud recall capability.

The extremely high precision score demonstrates that the model generated very few false positive alerts, making it highly suitable for real-world fraud monitoring environments where unnecessary transaction blocking can negatively affect customer experience and operational efficiency.

The model also maintained strong fraud detection capability with an ROC-AUC score of 0.970, demonstrating excellent ability to distinguish between fraudulent and legitimate transactions across classification thresholds.

---

# Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9533 | 0.7050 | 0.8991 | 0.7903 | 0.9638 |
| XGBoost | 0.9838 | 0.9643 | 0.8670 | 0.9130 | 0.9602 |
| Tuned XGBoost | 0.9870 | 1.0000 | 0.8670 | 0.9287 | 0.9698 |

---

# Model Interpretation

The final evaluation results demonstrated that the Tuned XGBoost model outperformed both Logistic Regression and the standard XGBoost model. Hyperparameter optimization significantly improved the model’s ability to balance fraud detection performance and false positive reduction.

Although Logistic Regression achieved slightly higher fraud recall, it produced substantially lower precision, resulting in more false positive fraud alerts. In contrast, Tuned XGBoost maintained strong fraud recall while dramatically improving precision and overall classification performance.

The superior performance of Tuned XGBoost highlights the importance of ensemble learning and hyperparameter optimization in fraud detection systems. The model successfully captured complex nonlinear fraud patterns, behavioral anomalies, and feature interactions that simpler models could not fully exploit.

---

# SHAP Explainability Findings

SHAP (SHapley Additive exPlanations) was implemented to improve model transparency and explain individual fraud predictions generated by the Tuned XGBoost model.

The SHAP analysis revealed the following major fraud drivers:

1. **Location Mismatch**  
   Transactions where customer location differed from transaction origin significantly increased fraud probability.

2. **Transaction Hour**  
   Transactions occurring during unusual hours contributed strongly to fraud predictions.

3. **Transaction Velocity (24-hour window)**  
   Rapid consecutive transactions within short time periods were strong indicators of suspicious activity.

4. **Corridor Risk**  
   High-risk geographic transaction corridors contributed positively to fraud likelihood.

5. **Transaction Fees and Exchange Rates**  
   Abnormal transaction fees and suspicious exchange rate behavior increased fraud probability.

6. **IP Risk Score**  
   Transactions associated with risky IP addresses were more likely to be fraudulent.

7. **New Device Usage**  
   Transactions originating from unfamiliar or untrusted devices contributed positively to fraud risk.

SHAP waterfall plots also provided transaction-level explainability by showing how individual feature contributions increased or decreased the final fraud prediction score represented as **f(x)**.

---

# Key Insights

1. **Location mismatch is a major fraud driver**

   Transactions where user location differed from transaction origin showed significantly higher fraud risk.

2. **Transaction behavior strongly predicts fraud**

   Transaction timing and rapid transaction velocity contributed heavily to fraud detection.

3. **Tuned XGBoost achieved the best overall performance**

   Hyperparameter optimization significantly improved precision, F1-score, and ROC-AUC performance, while Logistic Regression retained the highest fraud recall.

4. **Behavioral risk analytics improved fraud detection**

   Device trust, corridor risk, exchange behavior, and transaction timing improved fraud prediction capability.

5. **SHAP explainability improved model transparency**

   SHAP clearly explained why transactions were classified as fraudulent or legitimate, improving analyst trust and regulatory explainability.

6. **High precision significantly reduces false positives**

   Tuned XGBoost minimized unnecessary fraud alerts and reduced operational investigation burden, with a precision score of 1.0000 on the held-out test split.

---

# Business Recommendations Based on the Findings

## 1. Deploy Tuned XGBoost for Real-Time Fraud Monitoring

The Tuned XGBoost model should be deployed as the primary fraud detection engine due to its superior accuracy, precision, and overall fraud classification capability.
Fraud probability calibration and threshold tuning should be applied to balance false positives and missed fraud according to business risk tolerance.

## 2. Monitor Behavioral Fraud Indicators Continuously

NovaPay should prioritize continuous monitoring of:

- Transaction velocity
- Transaction timing
- Location mismatch
- Device trust signals
- Corridor risk indicators

## 3. Strengthen Device and IP Intelligence Systems

Integrate external IP reputation systems, geolocation analytics, and device fingerprinting solutions to improve fraud detection capability further.

## 4. Implement Explainable AI Dashboards

SHAP-based explainability dashboards should be integrated into fraud operations workflows to support:

- Fraud investigations
- Regulatory audits
- Analyst transparency
- Decision explainability

## 5. Continuously Retrain Fraud Models

Fraud patterns evolve continuously. NovaPay should periodically retrain and recalibrate fraud detection models using updated transaction data and emerging fraud behavior.

## 6. Optimize Fraud Thresholds Based on Business Priorities

Fraud thresholds should be adjusted depending on operational objectives:

- Higher precision reduces customer disruption
- Higher recall increases fraud detection coverage

Thresholds should be reviewed regularly with live production feedback to account for data drift and emerging fraud patterns.

---

# Conclusion

This project successfully developed a machine learning-based fraud detection framework capable of accurately identifying fraudulent digital money transfer transactions while minimizing false positive alerts.

Among all evaluated models, the Tuned XGBoost model achieved the best overall performance, delivering exceptional accuracy, precision, F1-score, and ROC-AUC performance. Logistic Regression retained the highest recall, highlighting the expected precision-recall tradeoff in fraud operations. The Tuned XGBoost model effectively captured complex fraud patterns and behavioral anomalies while maintaining strong fraud detection capability.

SHAP explainability further enhanced the system by providing transparent and interpretable fraud predictions. The analysis identified location mismatch, transaction timing, transaction velocity, corridor risk, and device intelligence as major fraud drivers.

Overall, the project provides NovaPay with a scalable, explainable, and production-ready fraud detection framework capable of supporting real-time financial risk monitoring, operational efficiency, regulatory compliance, and improved customer protection.


## Real-Time Demo App

The Streamlit deployment app uses the notebook's best-performing model (Tuned XGBoost) to score transactions in real time.

### Run Locally

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run dashboard/app.py
```

### Demo Capabilities

- Real-time single transaction fraud prediction with adjustable risk threshold.
- Batch CSV scoring with fraud probability, class prediction, and risk bands.
- In-app validation metrics and confusion matrix for stakeholder review.

## Fraud Detection API (FastAPI)

`inference.py` exposes the notebook's Tuned XGBoost model and preprocessing pipeline as a REST API for real-time, programmatic fraud scoring.

### Dependencies

The API requires `fastapi`, `pydantic`, and `uvicorn` in addition to the core project dependencies, all listed in `requirements.txt`:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run Locally

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn inference:app --reload
```

This starts the API at `http://127.0.0.1:8000`. `--reload` restarts the server automatically whenever `inference.py` changes.

### Making Predictions

**Interactive docs (easiest):** open `http://127.0.0.1:8000/docs`, expand `POST /predict`, click "Try it out," edit the example payload, and execute.
