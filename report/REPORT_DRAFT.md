# AI-Based Network Intrusion Detection System Using Machine Learning

## Abstract
This project develops a machine-learning-based network Intrusion Detection System (IDS) using the CIC-IDS2017 dataset. The system processes labeled network-flow records, cleans and transforms the data, removes constant and highly redundant features, and trains several machine-learning models for malicious-traffic detection. Binary classification is used to distinguish benign traffic from attacks, while multiclass classification is used to identify specific attack categories. Random Forest, XGBoost, Logistic Regression, and Isolation Forest are evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices. A Streamlit dashboard provides an interface for uploading network-flow CSV files and displaying detection results and alerts.

## 1. Dataset
CIC-IDS2017 contains network-flow records generated from normal activities and multiple attack scenarios. The supplied MachineLearningCSV archive contains eight CSV files and 78 measured traffic features plus a `Label` target.

The uploaded dataset contains approximately 2.83 million flows. The most frequent class is BENIGN, followed by DoS Hulk, PortScan, and DDoS. Several attack classes are rare, including Infiltration, SQL Injection, and Heartbleed.

## 2. Preprocessing
The CSV column names were stripped of surrounding whitespace. Labels were normalized to repair common encoding artifacts. Numeric features were converted to numeric values, infinite values were replaced by missing values, and missing numeric values were later imputed using medians learned from the training data only. Exact duplicates were removed within processing chunks for the working dataset.

Ten constant features were identified and removed. A further eleven highly redundant features were removed using a 0.995 correlation threshold calculated from the training data for the binary Random Forest experiment.

## 3. Working Dataset
Because the complete dataset is large, a stratified working sample was created with a maximum of 20,000 observations per class. This produced 107,641 flows across 15 labels. This sample was used for rapid model development while the original dataset was preserved unchanged.

## 4. Models
### Logistic Regression
Used as a simple linear baseline.

### Random Forest
Used as a primary supervised tree-based classifier. It also provides feature-importance estimates.

### XGBoost
Used as a gradient-boosted tree model for comparison.

### Isolation Forest
Used as an unsupervised anomaly-detection experiment. It was trained on benign training traffic and evaluated on mixed traffic.

## 5. Initial Binary Results
On the randomized working-sample benchmark, the models produced the following results:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9725 | 0.9886 | 0.9776 | 0.9831 | 0.9910 |
| Random Forest | 0.9986 | 0.9989 | 0.9993 | 0.9991 | 0.9998 |
| XGBoost | 0.9989 | 0.9990 | 0.9997 | 0.9993 | 1.0000 |
| Isolation Forest | 0.5560 | 0.9155 | 0.5028 | 0.6491 | 0.7446 |

These results are benchmark results on a randomized split and should not be interpreted as real-world deployment performance.

## 6. Multiclass Results
The multiclass Random Forest achieved approximately 99.11% accuracy and 0.90 macro F1 on the randomized working split. Common attack classes were detected strongly, while very rare classes such as Infiltration, SQL Injection, and XSS showed weaker class-specific recall/F1. This demonstrates the effect of severe class imbalance and limited examples.

## 7. Cross-Day Generalization
A stricter binary experiment trained on non-Friday files and tested on Friday files. Accuracy was approximately 52.3%, precision for attacks approximately 97.2%, attack recall approximately 30.6%, F1 approximately 46.6%, and ROC-AUC approximately 0.786.

This large reduction demonstrates an important limitation: the model can perform very well when similar attack patterns are present during training, but generalization to attack families or traffic patterns not represented during training is much harder.

## 8. Dashboard
The Streamlit dashboard allows a user to upload compatible network-flow CSV data. The application applies the saved preprocessing artifacts and Random Forest model, classifies each flow as BENIGN or ATTACK, calculates attack probability, displays summary statistics and alerts, and allows predictions to be downloaded as a CSV.

## 9. Limitations
- CIC-IDS2017 is a benchmark dataset and does not represent every real-world network.
- Some attack classes have very few examples.
- Randomized row splits can overestimate generalization because related flows may appear in both training and testing.
- The current dashboard performs file-based inference rather than direct live packet capture.
- Real deployment would require continuous traffic collection, robust drift monitoring, and additional validation on independent datasets.

## 10. Future Work
Future versions can add real-time packet/flow ingestion, independent validation datasets, better rare-class handling, threshold tuning, online learning, explainable alerts, and integration with SIEM or security-monitoring systems.
