# ECOM-MACHINE-LEARNING-MODE
## Overview

This project demonstrates a complete **time-series forecasting pipeline** for predicting product demand using Python and machine learning. The workflow includes data preprocessing, feature engineering, baseline forecasting models, a linear regression model, evaluation using RMSE, and multiple visualizations including a **smooth 3D demand surface**.

The goal is to understand how past sales and external factors such as price, promotions, holidays, and temperature influence future demand.

---

# Project Workflow

## 1. Data Loading and Sorting

The dataset is loaded from a CSV file and the date column is parsed as a datetime object.

The data is then sorted by:

* store_id
* product_id
* date

Sorting is essential in time-series modeling to ensure lag features are calculated correctly.

```
df = pd.read_csv("synthetic_demand_data.csv", parse_dates=["date"])
df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
```

---

# 2. Feature Engineering

Time-series models often rely on **lag features**, which represent past observations.

The following features are created:

### Lag Features

* **lag_1** → Sales from the previous day
* **lag_7** → Sales from one week earlier

### Rolling Feature

* **rolling_mean_7** → Average sales of the last 7 days (excluding today)

These features help capture demand trends and seasonality.

```
df["lag_1"] = df.groupby(["store_id","product_id"])["sales"].shift(1)
df["lag_7"] = df.groupby(["store_id","product_id"])["sales"].shift(7)

df["rolling_mean_7"] = (
    df.groupby(["store_id","product_id"])["sales"]
    .shift(1)
    .rolling(7)
    .mean()
)
```

Rows containing missing values created by lag operations are removed.

---

# 3. Time-Aware Train/Test Split

Instead of random splitting, the dataset is split using time order.

```
split_date = df["date"].quantile(0.8)

train = df[df["date"] < split_date]
test = df[df["date"] >= split_date]
```

This prevents **data leakage**, ensuring the model only learns from past data.

---

# 4. Evaluation Metric

Model performance is measured using **Root Mean Squared Error (RMSE)**.

```
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
```

RMSE penalizes large prediction errors more heavily and is commonly used in forecasting problems.

---

# 5. Baseline Forecast Models

Before training machine learning models, simple forecasting techniques are evaluated.

### Baselines Used

* Lag 1 Forecast
* Lag 7 Forecast
* Rolling Mean (7 days)

These serve as reference models to determine whether the ML model improves performance.

```
rmse_lag1 = rmse(test["sales"], test["lag_1"])
rmse_lag7 = rmse(test["sales"], test["lag_7"])
rmse_roll7 = rmse(test["sales"], test["rolling_mean_7"])
```

---

# 6. Correlation Analysis

A correlation matrix is generated to understand relationships between variables.

Features analyzed:

* sales
* lag_1
* lag_7
* rolling_mean_7
* price
* promotion
* holiday
* temperature

A heatmap helps identify which variables are most strongly related to sales.

---

# 7. Linear Regression Model

A machine learning model is trained using the engineered features.

### Features Used

* lag_1
* lag_7
* rolling_mean_7
* price
* promotion
* holiday
* temperature

```
model = LinearRegression()
model.fit(X_train, y_train)
```

Predictions are generated on the test dataset.

```
y_pred = model.predict(X_test)
rmse_lr = rmse(y_test, y_pred)
```

The model's RMSE is compared against baseline models.

---

# 8. Model Explainability

The regression coefficients help interpret how each feature affects sales.

Example interpretation:

* Positive coefficient → increases predicted sales
* Negative coefficient → decreases predicted sales

```
coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})
```

This helps understand demand drivers such as promotions or price changes.

---

# 9. Actual vs Predicted Visualization

A scatter plot compares predicted sales to actual sales.

This plot helps evaluate prediction accuracy and identify systematic errors.

```
plt.scatter(y_test[:1000], y_pred[:1000], alpha=0.3)
```

A good model produces points close to the diagonal.

---

# 10. 3D Demand Visualization (Scatter)

A 3D scatter plot shows the relationship between:

* lag_1
* price
* sales

```
ax.scatter(
    sample_3d["lag_1"],
    sample_3d["price"],
    sample_3d["sales"]
)
```

This provides an intuitive view of how price and past demand influence sales.

---

# 11. Smooth 3D Demand Surface

To visualize a continuous surface:

1. Create a grid of lag_1 and price values
2. Predict sales on that grid
3. Plot using a 3D surface

```
ax.plot_surface(
    lag1_grid,
    price_grid,
    sales_surface,
    cmap='viridis'
)
```

This produces a **smooth demand surface**.

Because the model is linear, the resulting surface appears as a **plane**.

---

# Key Learnings

This project demonstrates several important machine learning concepts:

* Time-series feature engineering
* Lag-based forecasting
* Avoiding data leakage
* Baseline model comparison
* Model interpretability
* Visual analysis of predictions
* 3D visualization of demand relationships

---

# Requirements

Install required libraries:

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

# Possible Improvements

This project can be extended by:

* Using **Polynomial Regression** for nonlinear demand curves
* Training **Random Forest or Gradient Boosting models**
* Adding **seasonality features**
* Implementing **cross-validation for time-series**
* Performing **hyperparameter tuning**

---

