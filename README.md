# **🔧 Predictive Maintenance System**

An end-to-end machine learning-based predictive maintenance system that analyzes machine sensor data to detect potential failures early, helping reduce unexpected downtime, maintenance costs, and machine failures.

# **🎯 Project Overview**

This project uses the AI4I 2020 Predictive Maintenance Dataset from Kaggle to build and evaluate machine learning models for early machine failure detection.

The project follows a complete ML pipeline covering EDA, data preprocessing, feature engineering, feature selection, imbalance handling, model comparison, and hyperparameter tuning.

# **🔎 Exploratory Data Analysis (EDA)**

A detailed EDA was performed to understand the dataset, identify patterns, detect anomalies, and prepare the data for modeling.

## **Key EDA & Preprocessing Steps**
**🧹 Data Cleaning**— Checked and handled data quality issues before modeling.
**📊 Distribution Analysis** — Used violin plots to understand feature distributions and identify potential anomalies.
**🔍 Failure Zone Analysis** — Used scatter plots to investigate machine failure patterns, particularly the relationship between Torque and Rotational Speed.
**🔗 Correlation Analysis** — Examined feature correlations and removed highly correlated/redundant features to reduce unnecessary information.
**📈 Outlier & Skewness Handling** — Rotational Speed contained numerous outliers and had a skewness of approximately 1.99. A log transformation was applied to reduce skewness and improve the feature distribution.
**⚖️ Normalization** — Applied MinMaxScaler to scale the selected numerical features to a common range.
**🧠 Feature Selection & Engineering**

After EDA, heavy feature selection was performed. Irrelevant features and highly correlated features were removed, retaining only the features that provided meaningful information for predicting machine failures.

Three additional domain-inspired features were engineered:

## **Feature	Description**
🌡️ **Temp_change** : Difference between process temperature and air temperature
⚙️ **Overstrain Index** :	Product of tool wear and torque
🔥 **Thermal Efficiency** : Temp_change divided by rotational speed

These engineered features capture relationships related to thermal conditions, mechanical stress, and machine operating conditions.

# **⚖️ Two Approaches to Class Imbalance**

Since machine failure is a relatively rare event, two different approaches were explored:

## **Version A — Class Weights**

Uses the original dataset while assigning higher weights to the minority failure class during model training.

## **Version B — SMOTE(Synthetic Over-Sampling)**

Uses SMOTE (Synthetic Minority Over-sampling Technique) on the training data to generate synthetic samples for the minority class.

This allowed a direct comparison between cost-sensitive learning and synthetic oversampling.

## **🤖 Model Development**

For both versions, multiple machine learning algorithms were trained and compared.

The best-performing models were subsequently hyperparameter tuned to optimize their performance.

The models were evaluated using metrics particularly relevant to imbalanced classification:

**Accuracy
Precision
Recall
F1-Score
PR-AUC
ROC-AUC*

## **🏆 Best Performing Model**

Across both approaches, **Random Forest (Version A — original data with class weights)* achieved the strongest overall results.

## **Metric	Score**
**📈 Training Score	99.31%
🎯 Testing Score	98.80%
Precision	~79%
Recall	~84%
F1-Score	~81%
PR-AUC	~83%
ROC-AUC	~98%*

## 💡 **Key takeaway**: The class-weighted Random Forest using the original dataset performed better overall than the SMOTE-based approach, while maintaining strong recall for detecting actual machine failures.

## 🔄 **End-to-End Pipeline**

                 AI4I Dataset
                      │
                      ▼
              Data Cleaning
                      │
                      ▼
              Exploratory Data
                 Analysis
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
  Distribution &              Correlation &
  Failure Analysis            Outlier Analysis
        │                           │
        └─────────────┬─────────────┘
                      ▼
             Feature Selection
                      │
                      ▼
            Feature Engineering
                      │
                      ▼
              Normalization
             (MinMaxScaler)
                      │
             ┌────────┴────────┐
             ▼                 ▼
        Version A          Version B
      Class Weights          SMOTE
             │                 │
             ▼                 ▼
       Multiple Models   Multiple Models
             │                 │
             └────────┬────────┘
                      ▼
             Hyperparameter
                  Tuning
                      │
                      ▼
               Model Evaluation
                      │
                      ▼
             🏆 Random Forest
              Version A

## 🚀 **Project Highlights**

🔬 Detailed exploratory data analysis

🧹 Data cleaning and preprocessing

📊 Violin plots and failure-zone analysis

🔗 Correlation-based feature reduction

📈 Log transformation for skewed Rotational Speed

⚖️ MinMaxScaler normalization

🧠 Extensive feature selection

🛠️ Domain-driven feature engineering

⚖️ Comparison of Class Weights vs. SMOTE

🤖 Multiple ML models

🎯 Hyperparameter tuning

📊 Imbalanced-class evaluation using Precision, Recall, F1, PR-AUC, and ROC-AUC

🏆 Best test score of 98.8% with Random Forest

## **🗂️ Dataset**

**Source*: AI4I 2020 Predictive Maintenance Dataset — Kaggle

The dataset contains machine operational and sensor measurements used to predict machine failure.

## **💡 Why Predictive Maintenance?**

**Unexpected machine failures can cause production downtime, increased maintenance costs, and reduced operational efficiency.*

**This project demonstrates how machine learning can transform machine sensor data into early failure warnings, enabling maintenance to be performed proactively rather than reactively.*

**⭐ If you find this project useful, consider starring the repository!**
