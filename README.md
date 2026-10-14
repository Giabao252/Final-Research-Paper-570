# Final-Research-Paper-570

Logistic Regression (Binomial) vs Gradient Boosting (ensemble method) in evaluating and classifying whether if a person has diabetes or not based on several factors in the Diabetes dataset (https://www.kaggle.com/datasets/datascikhan/diabetes?resource=download)

## What this project is

This is for the final research paper. The assignment is to pick two modeling approaches, run both on the same dataset, and compare them with visualizations. I picked logistic regression and gradient boosting, since one is easy to interpret and the other is usually better at squeezing out predictive accuracy, and I wanted to see how much that trade actually costs on this data.

The goal is to predict `diabetes_status` (0/1) from 12 lifestyle and demographic variables:

`bmi`, `age`, `systolic_bp`, `family_history_diabetes`, `cigarettes_per_day`, `alcohol_units_per_week`, `exercise_minutes_per_week`, `sedentary_hours_per_day`, `sleep_hours`, `stress_level`, `added_sugar_grams_per_day`, `fiber_grams_per_day`

I left out the lab values (`hba1c_percent`, the glucose columns), `diabetes_type`, and `diabetes_risk_score` because they pretty much determine the diagnostics.

