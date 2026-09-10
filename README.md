# Netflix Customer Retention & Churn Analysis

An end-to-end customer churn analytics project that combines predictive
modeling, customer segmentation, and moderation analysis to understand
which Netflix subscribers are likely to churn, why they churn, and how
retention strategies can be tailored to different customer groups.

## Business Problem

Customer retention is critical for subscription-based streaming platforms.
This project develops a unified analytical framework to answer three
questions:

1. Which behavioral and engagement factors are most predictive of churn?
2. Can subscribers be segmented into meaningful behavioral groups?
3. How do factors such as engagement, inactivity, subscription type,
   device, and age interact to influence churn?

## Dataset

The analysis uses a dataset of 5,000 Netflix subscribers containing
demographic, subscription, payment, device, and engagement information.
Explore the features for moderation analysisNetflix Consumer Churn dataset:
https://www.kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset
kaggle.com
Netflix Users Database:
https://www.kaggle.com/datasets/smayanj/netflix-users-databasekaggle.com


Key variables include:

- Watch hours
- Average watch time per day
- Days since last login
- Subscription type
- Payment method
- Device
- Favorite genre
- Number of profiles
- Customer churn status

## Analytical Approach

### 1. Logistic Regression

A logistic regression model was developed to predict customer churn and
identify the most influential churn drivers.

**Model Performance**

- Accuracy: 89.2%
- ROC-AUC: 96.6%
- Strong precision, recall, and F1 performance

Key churn signals included inactivity, engagement levels, subscription
characteristics, payment method, and device usage.

### 2. Customer Segmentation

K-Means clustering was used to identify behavioral customer segments.

Three interpretable customer personas emerged:

**At-Risk Newbies**
- Low engagement
- High churn probability
- Require early activation and re-engagement

**Stable Mainstream Viewers**
- Moderate and consistent engagement
- Lowest churn risk
- Benefit from continued value reinforcement

**Engaged Seniors**
- High engagement and strong viewing habits
- Moderate churn risk
- Benefit from personalized content and strategies to prevent content fatigue

### 3. Moderation Analysis

Moderation analysis was used to determine whether churn drivers affect
different customers differently.

Key interactions investigated included:

- Engagement × Inactivity
- Engagement × Subscription Type
- Device Type × Engagement
- Age × Watch Hours

A key finding was that engagement strongly reduces churn among active
customers, but its protective effect weakens as inactivity increases.

## Key Business Insights

- Inactivity is one of the strongest early indicators of churn.
- Higher engagement generally reduces churn risk.
- Churn drivers differ across customer segments.
- Engagement alone is not sufficient; recency and behavioral context matter.
- Customer segmentation makes churn predictions more actionable.
- Retention interventions should be personalized rather than applied uniformly.

## Strategic Recommendations

The analysis supports a targeted retention strategy based on both churn
risk and customer behavior:

- Prioritize customers in the highest predicted-risk groups.
- Trigger re-engagement campaigns when inactivity begins increasing.
- Personalize retention actions by customer segment.
- Use subscription and device context when designing interventions.
- Continuously update churn scores as new behavioral data becomes available.

## Repository Contents

- `data_processing.py` – Data preprocessing and churn modeling workflow
- `Netflix_Churn_Moderation_Analysis.ipynb` – Moderation analysis and
  interaction-effect exploration
- `README.md` – Project overview, methodology, results, and business insights

## Tools & Technologies

Python | Pandas | NumPy | Scikit-learn | Statsmodels | Matplotlib |
Seaborn | Logistic Regression | K-Means Clustering | Moderation Analysis

## Authors

Group 1 – DSBA 6276 Strategic Business Analytics  
University of North Carolina at Charlotte

