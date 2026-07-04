# PowerCo Customer Churn Prediction

Predicting churn among PowerCo's SME (small and medium enterprise) customers, completed as part of the **BCG Data Science Job Simulation** on Forage.

---

## Overview

PowerCo, a major gas and electricity utility, was losing SME customers and suspected that **price sensitivity** was the main reason. This project investigates why customers leave, identifies who is most at risk, and tests whether price is really the primary driver of churn.

Around **9.7% of PowerCo's SME customers churned** over the period studied, roughly 1 in 10 customers lost.

The three business questions guiding the work:
- Why are customers leaving?
- Which customers are most at risk?
- Is price sensitivity actually the main driver of churn?

---

## Approach

The project was built across four stages, each in its own notebook:

| Stage | What it covers |
|-------|----------------|
| **1. Exploratory Data Analysis** | Data cleaning, distribution and outlier checks, and seven business-focused visualisations comparing churned vs retained customers. |
| **2. Feature Engineering** | Built the modelling dataset (14,606 rows, 58 columns): extracted date-based tenure features, encoded categoricals, log-transformed skewed columns, and engineered price-difference features to test the price-sensitivity hypothesis. |
| **3. Modelling** | Trained a Random Forest classifier with `class_weight='balanced'` and tuned the decision threshold to prioritise catching at-risk customers. |
| **4. Executive Summary** | A single-slide stakeholder summary using the S-C-Q-R framework, translating technical results into business language. |

---

## Key Findings

- **Tenure is the strongest signal.** Customers in their first five years churn at **11.5%**, nearly double the **7%** rate for those with five to ten years. The early relationship is the most vulnerable period.
- **Price is *not* the primary driver.** Price-related features did appear in the model, but they were not among the top predictors, which directly answers PowerCo's original hypothesis.
- **Churned customers consumed less** electricity and gas on average, yet often carried a *higher* gross margin on power than those who stayed.

Tenure stood out even in the EDA, where it separated churned and retained customers clearly, and the model then confirmed it as one of the more important predictors. The pattern points to a specific problem: PowerCo is not holding on to its newer customers, who leave at a far higher rate than those past the five-year mark, so the first few years are where a retention strategy would have the most effect. Customers with more than ten years also churn at a slightly higher rate than the five-to-ten year group, which suggests PowerCo should tailor its retention approach across tenure bands.

When PowerCo framed the brief, they pointed to price sensitivity as the likely cause of churn. I treated that as a hypothesis to test rather than a conclusion to accept, and engineered price-difference features specifically so the model had a fair chance to pick it up. The model did use those features, but ranked them below other factors, so on this data it does not support price sensitivity as a main driver of churn, or as a sound basis for a retention strategy.

Testing a client's assumption before accepting it matters, because what we believe drives a business does not always show up in the numbers the way we expect. Getting the analysis right is what leads to the right strategy, and that has real financial consequences. This is still a first-pass model, so price should be investigated further before being ruled out entirely, but the early evidence points away from it.

---

## Model Results

The final Random Forest, evaluated with the threshold lowered to 0.2 to favour catching churners:

| Metric | Score |
|--------|-------|
| Recall (churn class) | 0.28 |
| Precision (churn class) | 0.30 |
| AUC-ROC | 0.66 |
| Accuracy | 86.3% |

The 86% accuracy looked strong at first, but I was cautious with it. Around 90% of the records are non-churn (retained) customers and only 10% are churners, so a model can reach high accuracy simply by predicting "retained" for almost everyone while completely failing to catch the churners we actually care about. Under this kind of class imbalance, accuracy is not a reliable measure on its own, and the right evaluation metric depends on the business problem being solved.

For PowerCo, the aim is to correctly identify churners so the company can build its retention strategy. A customer who is falsely predicted to stay but actually churns (a missed churn) is a greater loss than one who is flagged as a churn risk but would have stayed. That makes recall the primary metric for this problem.

Recall alone is not enough, though. Optimising purely for recall would push the model to label almost everyone as a churner, which destroys precision and wastes retention budget on customers who were never going to leave. Managing this precision-recall trade-off is a central part of the analysis, and I handled it by tuning the decision threshold. I chose 0.2, which raised recall from 0.05 to 0.28 while holding precision at around 0.30, without much further compromise. The F1 score peaked at this threshold, so given the data and methodology, 0.2 was the most balanced solution to the business problem.

This is a first-pass model rather than a finished product. The class imbalance limits performance, and the real value at this stage is the direction it gives PowerCo (focus on early-tenure customers, look beyond price). Given more time, the natural next steps would be resampling methods such as SMOTE, further hyperparameter tuning, and richer behavioural features.

---

## Repository Structure

```
powerco-churn-prediction/
├── 01_exploratory_data_analysis.ipynb
├── 02_feature_engineering.ipynb
├── 03_modelling.ipynb
├── client_data.csv
├── price_data.csv
├── clean_data_after_eda.csv
├── data_for_predictions.csv
└── README.md
```

---

## Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `Jupyter`

---

## Acknowledgements

Completed as part of the [BCG Data Science Job Simulation](https://www.theforage.com/simulations/bcg/data-science-ccdz/) on Forage. The dataset and brief were provided by the program; all analysis, code, and conclusions in this repository are my own.
