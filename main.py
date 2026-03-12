import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("synthetic_demand_data.csv", parse_dates=["date"])
df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)

print("Data Loaded:", df.shape)

df["lag_1"] = df.groupby(["store_id", "product_id"])["sales"].shift(1)
df["lag_7"] = df.groupby(["store_id", "product_id"])["sales"].shift(7)

df["rolling_mean_7"] = (
    df.groupby(["store_id", "product_id"])["sales"]
      .shift(1)
      .rolling(7)
      .mean()
)

df = df.dropna().reset_index(drop=True)

print("After feature engineering:", df.shape)

split_date = df["date"].quantile(0.8)

train = df[df["date"] < split_date]
test = df[df["date"] >= split_date]

print("Split date:", split_date)
print("Train shape:", train.shape)
print("Test shape:", test.shape)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_lag1 = rmse(test["sales"], test["lag_1"])
rmse_lag7 = rmse(test["sales"], test["lag_7"])
rmse_roll7 = rmse(test["sales"], test["rolling_mean_7"])

print("\nBaseline Results (RMSE)")
print("----------------------------")
print("Lag 1 RMSE:", rmse_lag1)
print("Lag 7 RMSE:", rmse_lag7)
print("Rolling 7 RMSE:", rmse_roll7)

results = {
    "Lag 1": rmse_lag1,
    "Lag 7": rmse_lag7,
    "Rolling 7": rmse_roll7
}

best_baseline_name = min(results, key=results.get)
best_baseline_value = results[best_baseline_name]

print("\nBest Baseline:", best_baseline_name)

corr = train[[
    "sales",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "price",
    "promotion",
    "holiday",
    "temperature"
]].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix (Train Data)")
plt.show()

features = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "price",
    "promotion",
    "holiday",
    "temperature"
]

X_train = train[features]
y_train = train["sales"]

X_test = test[features]
y_test = test["sales"]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse_lr = rmse(y_test, y_pred)

print("\nLinear Regression RMSE:", rmse_lr)
print("Best Baseline RMSE:", best_baseline_value)

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\nLinear Regression Coefficients")
print(coefficients)

plt.figure(figsize=(10, 6))
plt.scatter(y_test[:1000], y_pred[:1000], alpha=0.3)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Linear Regression: Actual vs Predicted (Sample)")
plt.show()

sample_3d = test.sample(3000, random_state=42)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    sample_3d["lag_1"],
    sample_3d["price"],
    sample_3d["sales"],
    c=sample_3d["sales"],
    cmap="viridis",
    alpha=0.6
)

ax.set_xlabel("Lag 1 Sales")
ax.set_ylabel("Price")
ax.set_zlabel("Sales")
ax.set_title("3D Demand Surface")

plt.show()

fixed_values = {
    "lag_7": test["lag_7"].mean(),
    "rolling_mean_7": test["rolling_mean_7"].mean(),
    "promotion": 0,
    "holiday": 0,
    "temperature": test["temperature"].mean()
}

lag1_range = np.linspace(test["lag_1"].min(), test["lag_1"].max(), 50)
price_range = np.linspace(test["price"].min(), test["price"].max(), 50)

lag1_grid, price_grid = np.meshgrid(lag1_range, price_range)

grid_df = pd.DataFrame({
    "lag_1": lag1_grid.ravel(),
    "lag_7": fixed_values["lag_7"],
    "rolling_mean_7": fixed_values["rolling_mean_7"],
    "price": price_grid.ravel(),
    "promotion": fixed_values["promotion"],
    "holiday": fixed_values["holiday"],
    "temperature": fixed_values["temperature"]
})

sales_pred_grid = model.predict(grid_df)

sales_surface = sales_pred_grid.reshape(lag1_grid.shape)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(
    lag1_grid,
    price_grid,
    sales_surface,
    cmap='viridis',
    alpha=0.8
)

ax.set_xlabel("Lag 1 Sales")
ax.set_ylabel("Price")
ax.set_zlabel("Predicted Sales")
ax.set_title("Smooth 3D Demand Surface (Linear Regression)")

fig.colorbar(surface, shrink=0.5, aspect=5)

plt.show()