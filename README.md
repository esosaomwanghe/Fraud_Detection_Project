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