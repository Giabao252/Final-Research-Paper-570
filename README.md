# Final-Research-Paper-570

Logistic Regression (Binomial) vs Gradient Boosting (ensemble method) in evaluating and classifying whether if a person has diabetes or not based on several factors in the Diabetes dataset (https://www.kaggle.com/datasets/datascikhan/diabetes?resource=download)

## What this project is

This is for the final research paper. The assignment is to pick two modeling approaches, run both on the same dataset, and compare them with visualizations. I picked logistic regression and gradient boosting, since one is easy to interpret and the other is usually better at squeezing out predictive accuracy, and I wanted to see how much that trade actually costs on this data.

The goal is to predict `diabetes_status` (0/1) from 12 lifestyle and demographic variables:

`bmi`, `age`, `systolic_bp`, `family_history_diabetes`, `cigarettes_per_day`, `alcohol_units_per_week`, `exercise_minutes_per_week`, `sedentary_hours_per_day`, `sleep_hours`, `stress_level`, `added_sugar_grams_per_day`, `fiber_grams_per_day`

I left out the lab values (`hba1c_percent`, the glucose columns), `diabetes_type`, and `diabetes_risk_score` because they pretty much determine the diagnostics.

## Where things are

| File | What it does |
| --- | --- |
| `data_processing.ipynb` | Cleans the data, encodes the predictors, makes a stratified 80/20 train/test split, imputes and scales (fitted on train only), and saves everything to `data/` |
| `logistic_regression.ipynb` | Fits and tunes the logistic regression, evaluates it on the test set, and builds the odds ratio table and plots |
| `gradient_boosting.ipynb` | Fits and tunes the gradient boosting model, evaluates it, and compares it head to head with logistic regression |
| `data/` | Saved splits (parquet), the fitted imputer and scaler, and each model's saved outputs |
| `figures/` | Plots for the paper |

## How to run it

1. Download the CSV and drop it in the project root.
2. Run `data_processing.ipynb` top to bottom. If you get a pyarrow error on the save cell, restart the kernel and run it again from the top. The pip install and the pandas import in the same session confuse it.
3. Run `logistic_regression.ipynb`, then `gradient_boosting.ipynb`. The second one reads the first one's saved predictions for the comparison section, and its hyperparameter search takes about 5 minutes.

Both models read the same saved splits, so the comparison is fair. Everything uses `random_state=42`, and all tuning (including picking the decision threshold) happens on the training set only. The test set is only used for the final numbers.

## The data

- About 2% of rows had no `diabetes_status`, so they were dropped, leaving 97,980 rows (78,384 train, 19,596 test). The classes are roughly 72% no diabetes and 28% diabetes.
- Missing values in the predictors are only 2-5% per column, and they show up at about the same rate in both outcome groups, so median imputation seemed safe. Gradient boosting skips imputation and handles the NaNs itself.
- The predictors aren't badly correlated with each other (highest VIF is 1.69), so the logistic regression coefficients are stable.

## Results

Everything below is on the held-out test set.

### Headline numbers

| | Logistic regression | Gradient boosting |
| --- | --- | --- |
| ROC-AUC | **0.781** (95% CI 0.773 to 0.789) | 0.778 (95% CI 0.770 to 0.787) |
| PR-AUC | **0.667** | 0.666 |
| Accuracy at 0.5 cutoff | 0.800 | 0.799 |
| Precision / recall at 0.5 | 0.733 / 0.448 | 0.734 / 0.442 |
| Best cutoff (picked on train) | 0.387 | 0.297 |
| Precision / recall at best cutoff | 0.605 / 0.595 | 0.578 / 0.621 |
| F1 at best cutoff | **0.600** | 0.599 |

The prevalence is 28%, so a model that knew nothing would sit at a PR-AUC of 0.28. Both are well above that, but an AUC around 0.78 is decent, not great.

### Is one actually better?

Not really. Logistic regression comes out ahead by 0.003 AUC. A paired bootstrap (both models scored on the same 1,000 resamples of the test set) puts the difference at -0.0026 for gradient boosting, with a 95% interval of -0.0044 to -0.0008. That interval doesn't include zero, so the gap probably isn't pure noise, but it's far too small to matter in practice. The ROC and precision-recall curves lie almost on top of each other.

So the trade I was curious about turned out to cost nothing. The extra flexibility of boosting didn't buy any accuracy, which makes sense once you look at the data: the relationships are mostly smooth and one-directional (more BMI, more age, more risk), and a logistic regression handles that fine.

### Where they do differ

- **Calibration.** Gradient boosting's predicted probabilities line up with what actually happens better than logistic regression's do. Logistic regression under-predicts in the 0.2 to 0.4 range and over-predicts at the high end. This is the one clear win for boosting.
- **Why that happens.** The partial dependence plot shows BMI risk is curved, not a straight line: predicted risk goes from about 15% at a BMI of 19 to about 60% at a BMI of 39, and it speeds up past 30. Logistic regression is forced to treat BMI as a straight line on the log-odds scale, which is probably what's causing the calibration miss. Adding a BMI squared or spline term to the logistic regression is the obvious next thing to try.
- **Interpretability.** Logistic regression gives a direct answer to "how much does each factor matter and in which direction", with confidence intervals. Boosting needs permutation importance and partial dependence plots to get a similar picture, and even then you get shapes, not a single number per variable.

### Which factors matter

Both models agree on the ranking, which is reassuring.

| Factor | Logistic regression (odds ratio per 1 SD) | Gradient boosting (AUC drop when shuffled) |
| --- | --- | --- |
| BMI | 2.05 | 0.091 |
| Age | 1.62 | 0.037 |
| Family history | 1.49 | 0.029 |
| Systolic blood pressure | 1.29 | 0.013 |
| Cigarettes per day | 1.13 | 0.003 |
| Sleep hours | 0.91 | 0.002 |
| Exercise minutes | 0.91 | 0.001 |
| Fiber | 0.93 | 0.001 |
| Stress level | 1.00 (p = 0.89) | 0.000 |
| Added sugar | 0.99 (p = 0.17) | 0.000 |
| Alcohol | 1.03 | 0.000 |

BMI is far and away the strongest signal. Exercise, sleep, and fiber look mildly protective in the logistic regression. Stress level and added sugar do nothing in either model, which is a bit surprising and might say more about how this dataset was put together than about real biology. In plain units, each extra BMI point multiplies the odds of diabetes by 1.13, and having a family history multiplies them by 2.23.

### About the cutoff

At the default 0.5 cutoff both models only catch about 44-45% of the people who have diabetes. That's because the classes are imbalanced. Moving the cutoff down (0.387 for logistic regression, 0.297 for boosting, both chosen using the training data only) gets precision and recall both to around 0.6 and bumps F1 from about 0.55 to 0.60. Which cutoff is "right" depends on whether missing a case or flagging a healthy person is the worse mistake, which is a judgment call for the paper's discussion section.

### Plots

All in `figures/`:

- `comparison_roc_pr_calibration.png`: ROC, precision-recall, and calibration for both models side by side
- `lr_odds_ratios.png`: odds ratio forest plot
- `gb_permutation_importance.png`, `gb_partial_dependence.png`: boosting's version of "what matters and how"
- `lr_roc_pr.png`, `gb_roc_pr.png`, `lr_confusion_matrices.png`, `gb_confusion_matrices.png`, `lr_calibration.png`: each model on its own

## Caveats

- One train/test split, so the numbers have some luck baked in. The bootstrap intervals capture noise in the test set, not noise from which rows landed in train.
- Only 12 predictors, chosen to be things a person could know without a lab test. That's why the AUC tops out near 0.78. Adding lab values would push it way up, but it would be almost like predicting diabetes from the diagnosis.
- I only tried a plain logistic regression. No interaction terms, no splines. Boosting could look better against a stronger linear baseline, or worse.

## Still to do

- Try a BMI squared or spline term in the logistic regression and see if the calibration gap closes.
- Write the paper (3,000 to 5,000 words, PDF).
