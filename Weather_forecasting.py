import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

#load dataset
data = pd.read_csv("popular_cities_weather.csv")
print("dataset.shape:")
print(data.shape)
print("\ncolumns:")
print(data.columns)
print("\nFirst 5 rows:")
print(data.head())
print("\nmissing values:")
print(data.isnull().sum())
print("\nnumber of cities:")
print(data['city'].nunique())
print("\nsample cities:")
print(data['city'].unique()[:20])

data['date'] = pd.to_datetime(data['date'])
data = data.drop(columns = ['wspd','tsun'])
data['tavg'] = data['tavg'].fillna(data['tavg'].median())
data['tmax'] = data['tmax'].fillna(data['tmax'].median())
data['tmin'] = data['tmin'].fillna(data['tmin'].median())
data['prcp'] = data['prcp'].fillna(0)
data['pres'] =data['pres'].fillna(data['pres'].median())
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month

print(data.head())
print("\nmissing values after cleaning:")
print(data.isnull().sum())
data_original = data.copy()
data = data.sort_values(['city','date'])

data['next_month_temp'] =(
    data.groupby('city')['tavg'].shift(-1)
)
data = data.dropna(subset=['next_month_temp'])

data = pd.get_dummies(data,columns=['city'])
train_data = data[data['year'] <= 2024]
test_data = data[data['year'] == 2025]

X_train = train_data.drop(
    columns=['next_month_temp', 'date']
)

y_train = train_data['next_month_temp']

X_test = test_data.drop(
    columns=['next_month_temp', 'date']
)

y_test = test_data['next_month_temp']

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("MAE:", round(mae, 2))
print("R2:", round(r2, 3))

future_data = data_original[
    (data_original['year']==2025) &
    (data_original['month']==12)
].copy()

print("Cities available for forecast:",len(future_data))
forecast_results =[]

for month in range(1,13):
    temp_month = future_data.copy()
    
    temp_month['year'] = 2026
    temp_month['month'] = month

    temp_month_encoded = pd.get_dummies(
        temp_month,
        columns=['city']
    )
    temp_month_encoded = temp_month_encoded.reindex(
        columns=X_train.columns,
        fill_value=0
    )
    
    predicted_temp = model.predict(temp_month_encoded)
    
    result = pd.DataFrame(
        {'city': future_data['city'],
         'forecast_month':month,
         'predicted_temp': predicted_temp
        }
    )
    forecast_results.append(result)
forecast_2026 = pd.concat(
    forecast_results,
    ignore_index=True
)
def classify_weather(temp):
    if temp >= 35:
        return "Very hot"
    elif temp >=28:
        return "Sunny"
    elif temp >= 20:
        return "Pleasant"
    else:
        return "Cold"
    
forecast_2026['weather_condition'] =(
    forecast_2026['predicted_temp']
    .apply(classify_weather)
)    
    

print("\nForecast Sample:")
print(
    forecast_2026[
        [
            'city',
            'forecast_month',
            'predicted_temp',
            'weather_condition'
        ]
    ].head(20)
)

forecast_2026.to_csv(
    "weathery_forecast_2026.csv",
    index=False

)
print("Forecast saved successfully!")

