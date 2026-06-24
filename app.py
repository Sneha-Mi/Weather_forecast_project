import pandas as pd
import streamlit as st
st.title("AI Weather Forecasting System")
st.markdown("""Predict future weather conditions for Indian cities using Machine Learning. 
           
            Select a city and month to view temperature forecasts and weather trends.""" )

forecast = pd.read_csv("weathery_forecast_2026.csv")
print("Number of cities:",forecast["city"].nunique())

st.set_page_config(
    page_title = "Weather Forecasting System",
    page_icon="🌤️",
    layout ="wide"
    )    
city = st.selectbox(
    " 🏙️ Select City",
    sorted(forecast["city"].unique())
)
city_data = forecast[
    forecast["city"] == city
].sort_values("forecast_month")





#forecast = pd.read_csv("weathery_forecast_2026.csv")



month = st.selectbox(
    " 📅 Select Month",
    range(1, 13)
)

result = forecast[
    (forecast["city"] == city)
    & (forecast["forecast_month"] == month)
]

if not result.empty:

    temp = result.iloc[0]["predicted_temp"]
    weather = result.iloc[0]["weather_condition"]

    st.subheader(f"Forecast for {city}")

    st.write(f"Month: {month}")
    st.write(f"Temperature: {temp:.2f} °C")
    st.write(f"Condition: {weather}")
    st.subheader("Temperature Trend for 2026")

    city_data = forecast[
        forecast["city"] == city
    ].sort_values("forecast_month")

    st.line_chart(
        city_data.set_index("forecast_month")["predicted_temp"]
    )