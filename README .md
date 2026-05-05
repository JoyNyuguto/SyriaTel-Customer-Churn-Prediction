# SyriaTel Customer Churn Prediction

**Author:** Joy Wambui Nyuguto  
**Dataset:** [SyriaTel Churn — Kaggle](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset)

---

## Overview

SyriaTel is a telecommunications company losing revenue to customer churn. This project builds a binary classification model to predict which customers are likely to leave, giving the retention team a chance to intervene before they do.

The final model — a **Tuned Decision Tree** — achieves **66% recall** and **0.707 F1-score** on the held-out test set across five model iterations.

---

## Business and Data Understanding

**Stakeholder:** SyriaTel's Customer Retention Team  
**Problem:** Customers churn without warning. Retaining an existing customer costs 5–7× less than acquiring a new one, so early identification of at-risk customers is critical.  
**Why ML:** With 20+ features and complex interactions between them, a classification model learns patterns no simple rule can capture.

**Dataset:**
- 3,333 customer records, 21 features
- Target: `churn` (True/False) — binary classification
- Churn rate: 14.5% — class-imbalanced dataset
- No missing values

**Features used:** Account length, plan types, daytime/evening/night/international usage minutes and calls, voicemail messages, customer service calls.

**Limitations:** Trained on one historical snapshot. Missing actual bill amounts and complaint categories which would likely strengthen predictions.

---

## Exploratory Data Analysis

### Chart 1 — Customer Service Calls vs Churn
![Service Calls](images/chart2_service_calls_churn.png)

> Customers who called support **4 or more times** churn at over 40% — nearly three times the average rate. Repeated calls signal unresolved dissatisfaction.

---

### Chart 2 — Total Day Minutes vs Churn
![Day Minutes](images/chart3_day_minutes_churn.png)

> Churned customers have a median of **215 day minutes** vs **178** for retained customers. The extra 37 minutes per day means a consistently higher bill — a key driver of churn.

---

### Chart 3 — Plan Type vs Churn Rate
![Plan Types](images/chart4_plan_churn.png)

> International plan subscribers churn at **42%** — nearly 4× the average. Voice mail subscribers churn at just **8.7%**, below average.

---

## Modeling

Five models were built iteratively:

| Model | Test Recall | Test F1 | Test Acc |
|-------|------------|---------|---------|
| Logistic Regression (Baseline) | 0.216 | 0.327 | 0.856 |
| Decision Tree (Default) | 0.711 | 0.697 | 0.912 |
| **Decision Tree (Tuned) ← Final** | **0.660** | **0.707** | **0.921** |
| Decision Tree + SMOTE | 0.711 | 0.570 | 0.844 |
| Random Forest + SMOTE | 0.546 | 0.599 | 0.894 |

The **Tuned Decision Tree** was selected as the final model based on the highest F1-score (0.707), which best balances catching churners against minimising false alarms.

---

## Evaluation

### Model Comparison
![Model Comparison](images/model_comparison.png)

> The Tuned Decision Tree achieves the best F1-score across all five models.

---

### ROC Curves — All Models
![ROC Curves](images/roc_curves_comparison.png)

> The Tuned Decision Tree has the highest AUC (0.834), confirming it is the best model at distinguishing churners from non-churners across all thresholds.

---

### Feature Importances
![Feature Importances](images/feature_importances.png)

> **Total day minutes (0.30)**, **customer service calls (0.16)**, and **total eve minutes (0.14)** are the top three drivers of churn — matching the patterns found in EDA.

---

## Conclusion

The Tuned Decision Tree successfully identifies customers at risk of churning with a recall of 0.660 and F1-score of 0.707 across five model iterations. Total day minutes, customer service calls, and evening usage are the strongest predictors — insights SyriaTel can act on immediately.

---

## Recommendations

1. **Flag customers after 3 service calls** — churn jumps to 40%+ at call 4
2. **Audit the international plan** — 42% churn rate signals a pricing problem
3. **Send billing alerts for users over 215 day minutes/day** — prevent bill shock
4. **Promote the voice mail plan** to at-risk customers — churn rate is only 8.7%
5. **Run the model monthly** — score all customers and target top 20% at-risk

---

## Next Steps

1. Collect richer data — actual bill amounts, complaint categories, contract tenure
2. Retrain the model quarterly to keep up with shifting customer behaviour
3. Explore XGBoost for potential performance gains beyond the current F1 of 0.707

---


