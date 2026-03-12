import pandas as pd
import numpy as np

np.random.seed(42)

num_stores = 20
num_products = 50
years = 2

dates = pd.date_range(start="2020-01-01", periods=365*years)

data = []

for store in range(1, num_stores + 1):
    store_factor = np.random.uniform(0.8, 1.2)
    
    for product in range(1, num_products + 1):
        base_demand = np.random.uniform(20, 100)
        base_price = np.random.uniform(10, 50)
        
        for date in dates:
            month = date.month
            day_of_week = date.weekday()
        
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * month / 12)

            weekend_factor = 1.1 if day_of_week >= 5 else 1
            
            promotion = np.random.choice([0, 1], p=[0.85, 0.15])
            promo_factor = 1.3 if promotion == 1 else 1
            
            holiday = np.random.choice([0, 1], p=[0.97, 0.03])
            holiday_factor = 1.2 if holiday == 1 else 1

            temperature = 20 + 10 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 2)
            
            price = base_price * np.random.uniform(0.9, 1.1)
            
            price_effect = 1 - (price - base_price) / base_price
            
            sales = (
                base_demand
                * store_factor
                * seasonal_factor
                * weekend_factor
                * promo_factor
                * holiday_factor
                * price_effect
                + np.random.normal(0, 5)
            )
            
            sales = max(0, int(sales))  
            
            data.append([
                date,
                store,
                product,
                price,
                promotion,
                holiday,
                temperature,
                sales
            ])

columns = [
    "date",
    "store_id",
    "product_id",
    "price",
    "promotion",
    "holiday",
    "temperature",
    "sales"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("synthetic_demand_data.csv", index=False)

print("Dataset generated!")
print(df.shape)
