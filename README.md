# SyriaTel Customer Churn Prediction

**Author:** Joy Wambui Nyuguto  
**Dataset:** [SyriaTel Churn — Kaggle](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset)

---

## Overview

This project presents a full data science analysis to guide SyriaTel's Customer Retention Team in identifying customers who are likely to churn. Using historical customer usage and account data, we built a binary classification model that predicts churn risk — giving the retention team a chance to intervene before a customer leaves.

The final model — a **Tuned Decision Tree** — achieves **66% recall** and **0.707 F1-score** on the held-out test set across five model iterations.

---

## Business Understanding

### Key Stakeholder
SyriaTel's Customer Retention Team

### Business Problem
SyriaTel is losing revenue every time a customer churns. Acquiring a new customer costs 5–7× more than retaining an existing one, making early identification of at-risk customers critical. Without a data-driven approach, the retention team has no way to prioritise outreach efficiently.

### Key Business Questions

1. **How severe is the churn problem — and how imbalanced is our data?**  
   Understanding the proportion of churners determines whether standard accuracy is a reliable metric.

2. **Does poor customer service drive customers to leave?**  
   If repeated service calls signal unresolved dissatisfaction, the team can intervene before a customer reaches breaking point.

3. **Are high-usage customers more likely to churn due to billing pressure?**  
   Heavy users generate more revenue but may face bill shock — identifying this group helps SyriaTel protect its most valuable segment.

4. **Does the type of subscription plan predict churn risk?**  
   If certain plans are linked to higher churn, SyriaTel can re-evaluate pricing or the customer experience tied to those plans.

5. **Which customer attributes show the strongest relationship with churn?**  
   Understanding feature correlations helps identify the strongest predictors and guides model feature selection.

---

## Data Understanding and Analysis

### Source of Data

The analysis uses one CSV dataset sourced from Kaggle:

| Source | File | Size |
|--------|------|------|
| Kaggle — SyriaTel Churn | `bigml_59c28831336c6604c800002a.csv` | 3,333 rows, 21 columns |

### Description of Data

**`bigml_59c28831336c6604c800002a.csv`** contains historical customer records from SyriaTel including account details, usage patterns, plan subscriptions, and a churn label. It is the sole dataset used for all analysis, feature engineering, model training, and evaluation.

| Feature Category | Columns |
|-----------------|---------|
| Account info | `state`, `account length`, `area code` |
| Plan type | `international plan`, `voice mail plan` |
| Daytime usage | `total day minutes`, `total day calls`, `total day charge` |
| Evening usage | `total eve minutes`, `total eve calls`, `total eve charge` |
| Night usage | `total night minutes`, `total night calls`, `total night charge` |
| International usage | `total intl minutes`, `total intl calls`, `total intl charge` |
| Service history | `number vmail messages`, `customer service calls` |
| **Target** | **`churn`** (True/False — binary classification) |

**Key properties:**
- 3,333 customer records, 21 features, no missing values
- Churn rate: 14.5% — class-imbalanced dataset
- Charge columns were dropped during preparation as they are perfectly derived from minutes × rate (redundant)

**Limitations:** The dataset is a single historical snapshot. It does not include actual bill amounts, complaint categories, or contract tenure — all of which would likely strengthen predictions.

---

### Visualizations

#### Visualization 1 — Customer Service Calls vs Churn
![Customer Service Calls vs Churn](images/chart2_service_calls_churn.png)

Customers who called support **4 or more times** churn at over 40% — nearly three times the average rate of 14.5%. The churn rate rises sharply from call 3 onwards, suggesting that unresolved issues escalate quickly into a decision to leave. This makes customer service call count one of the most actionable predictors in the dataset.

---

#### Visualization 2 — Total Day Minutes vs Churn
![Total Day Minutes vs Churn](images/chart3_day_minutes_churn.png)

Churned customers have a median of **215 day minutes** compared to **178 minutes** for retained customers — a gap of 37 minutes per day. Since SyriaTel charges by the minute, heavier users accumulate higher bills, which appears to drive dissatisfaction and eventually churn.

---

#### Visualization 3 — Plan Type vs Churn Rate
![Plan Type vs Churn Rate](images/chart4_plan_churn.png)

International plan subscribers churn at **42%** — nearly four times the overall average. In contrast, voice mail plan subscribers churn at just **8.7%**, well below average. This suggests the international plan may have pricing or quality issues, while the voice mail plan is associated with higher engagement and retention.

---

## Modeling

Five models were built iteratively, each justified by the results of the previous:

| Model | Test Recall | Test F1 | Test Acc |
|-------|------------|---------|---------|
| Logistic Regression (Baseline) | 0.216 | 0.327 | 0.856 |
| Decision Tree (Default) | 0.711 | 0.697 | 0.912 |
| **Decision Tree (Tuned) ← Final** | **0.660** | **0.707** | **0.921** |
| Decision Tree + SMOTE | 0.711 | 0.570 | 0.844 |
| Random Forest + SMOTE | 0.546 | 0.599 | 0.894 |

The **Tuned Decision Tree** was selected as the final model based on the highest F1-score (0.707), which best balances catching churners against minimising false alarms. SMOTE improved recall but reduced precision too much, dropping overall F1. Random Forest with SMOTE performed worse on both key metrics.

---

## Evaluation

#### Model Comparison — Recall & F1-Score
![Model Comparison](images/model_comparison.png)

The Tuned Decision Tree achieves the best F1-score (0.707) across all five models, confirming it as the strongest overall performer.

---

#### ROC Curves — All Models
![ROC Curves](images/roc_curves_comparison.png)

The Tuned Decision Tree has the highest AUC of **0.834**, meaning it is the best model at distinguishing churners from non-churners across all possible thresholds. All models sit well above the 0.5 random baseline.

---

#### Feature Importances — Top Drivers of Churn
![Feature Importances](images/feature_importances.png)

**Total day minutes (0.30)** is the single strongest predictor of churn. **Customer service calls (0.16)** ranks second, and **total eve minutes (0.14)** third — all consistent with the patterns identified during EDA.

---

## Conclusions

Based on the analysis, SyriaTel should:

- **Flag customers after their 3rd service call** — churn jumps above 40% at call 4, making this the clearest early warning signal in the dataset
- **Audit the international plan urgently** — a 42% churn rate signals a serious pricing or quality problem that is costing SyriaTel customers
- **Send billing alerts for heavy daytime users** — customers using over 215 day minutes per day are at the highest churn risk due to bill shock
- **Promote the voice mail plan to at-risk customers** — subscribers churn at only 8.7%, suggesting this plan improves engagement and retention
- **Deploy the model monthly** — score all active customers for churn risk and focus retention outreach on the top 20% highest-risk accounts

---

## Next Steps

1. **Collect richer data** — add actual monthly bill amounts, complaint categories, and contract tenure to strengthen model predictions
2. **Retrain the model quarterly** — customer behaviour shifts over time and the model needs regular updates to stay accurate
3. **Explore XGBoost** — evaluate gradient boosting for potential performance gains beyond the current F1 of 0.707

---



