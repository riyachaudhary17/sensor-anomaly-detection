# Sensor Anomaly Detection

Predictive modeling project (Celebal Technologies Kaggle Challenge) to detect anomalies in industrial sensor readings from a highly imbalanced time-series dataset.

## Problem

Given readings from 5 sensors (X1–X5) captured over time, predict whether a given reading is a normal observation (0) or an anomaly (1). Anomalies made up less than 1% of the data, making this a severe class-imbalance problem.

## Dataset

- **Train:** 1,639,424 rows → 1,582,109 after removing 57,315 duplicates
- **Test:** 409,856 rows
- **Features:** 5 continuous sensor readings (X1–X5) + timestamp (`Date`)
- **Target distribution:** ~99.14% normal vs ~0.86% anomaly

## Approach

1. **EDA** — checked missing values, class distribution, per-sensor distributions by class (KDE plots), boxplots, IQR-based outlier summary, correlation heatmap, and anomaly rate over time.
2. **Preprocessing** — median imputation for missing sensor values, duplicate removal, outlier capping using a widened IQR bound (3×IQR) to avoid clipping genuine anomalies.
3. **Feature Engineering** (33 features total) —
   - Time features: hour, day, day of week, month
   - Rolling statistics (window=5): rolling mean & std per sensor
   - Lag features: previous reading and delta from previous reading per sensor
   - Cross-sensor features: X1/X2 ratio, X4−X5 difference, sensor sum, sensor std
4. **Modeling** — trained and compared 8 models: Logistic Regression, KNN, Decision Tree, SVM (RBF), Random Forest, XGBoost, LightGBM, CatBoost, and a Keras Neural Network (with `class_weight` / `scale_pos_weight` to handle imbalance).
5. **Hyperparameter tuning** — `GridSearchCV` (3-fold stratified CV, scoring=F1) on XGBoost across `max_depth`, `n_estimators`, `learning_rate`.
6. **Validation** — stratified train/validation split (80/20) plus a separate **time-based backtest** (train on first 80% of the timeline, test on the last 20%) and residual analysis to check for systematic prediction drift over time.

## Tools & Libraries

Python, Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM, CatBoost, TensorFlow/Keras, Matplotlib, Seaborn

## Results

| Model | Accuracy | Precision (Anomaly) | Recall (Anomaly) | F1 (Anomaly) |
|---|---|---|---|---|
| Logistic Regression | 0.90 | 0.07 | 0.89 | 0.13 |
| KNN | 0.99 | 0.69 | 0.22 | 0.33 |
| Decision Tree | 0.90 | 0.08 | 0.90 | 0.14 |
| SVM (RBF) | 0.96 | 0.13 | 0.66 | 0.22 |
| Random Forest | 0.96 | 0.16 | 0.77 | 0.26 |
| XGBoost | 0.94 | 0.12 | 0.93 | 0.21 |
| CatBoost | 0.92 | 0.09 | 0.94 | 0.17 |
| Neural Network | 0.91 | 0.08 | 0.93 | 0.15 |
| **XGBoost (Tuned)** | **0.97** | **0.24** | **0.83** | **0.37** |

**Best model:** Tuned XGBoost (`max_depth=8, n_estimators=300, learning_rate=0.1`), selected on F1-score — best precision/recall trade-off for the minority (anomaly) class.

On a time-based backtest (training on the earliest 80% of the timeline and testing on the most recent 20%), the tuned model held up reasonably well (accuracy 0.98) but recall on anomalies dropped to 0.56, indicating some distribution drift over time — a realistic limitation worth flagging rather than hiding.

## Key Learnings

- With ~99% class imbalance, accuracy is a misleading metric — F1 and recall on the minority class matter far more.
- Rolling/lag features and cross-sensor interactions gave a meaningful lift over raw sensor values alone.
- Time-based backtesting revealed a real-world validity gap that a random train/val split had missed — the model generalizes less well to unseen future time periods, likely because anomaly patterns are not fully stationary.
- Threshold/hyperparameter tuning targeted at F1 (rather than accuracy) was essential to get a usable anomaly detector.

## Files

- `notebook.ipynb` — full analysis, feature engineering, modeling, and evaluation
- `submission.csv` — final predictions on the test set
