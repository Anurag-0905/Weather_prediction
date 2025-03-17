#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# In[2]:


def fetch_weather_data(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    return{
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"]
    }


# In[3]:


api_key = "2ecf5ce0116753bbefbb61ba24047c1d"
city = "Delhi"
weather_data = fetch_weather_data(api_key, city)
print("Current Weather Data:")
print(f"Temperature: {weather_data['temperature']} °C")
print(f"Humidity: {weather_data['humidity']} %")
print(f"Pressure: {weather_data['pressure']} hPa")


# In[10]:


data = pd.DataFrame({
    "temperature": [30, 32, 35, 28, 25],
    "humidity": [70, 65, 60, 80, 85],
    "pressure": [1012, 1010, 1008, 1015, 1018],
    "future_temp": [31, 33, 34, 29, 26],
    "future_humidity": [68, 64, 59, 79, 83],
    "future_pressure": [1013, 1011, 1009, 1016, 1017] 
})


# In[11]:


X = data[["temperature", "humidity", "pressure"]]
y = data[["future_temp", "future_humidity", "future_pressure"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultiOutputRegressor(LinearRegression())
model.fit(X_train, y_train)


# In[12]:


y_pred = model.predict(X_test)
print("Predicted Future Weather Condition:")
print(f"Temperature: {y_pred[:, 0]}")
print(f"Humidity: {y_pred[:, 1]}")
print(f"Pressure: {y_pred[:, 2]}")


# In[13]:


mae_temp = mean_absolute_error(y_test.iloc[:, 0], y_pred[:, 0])
mae_humidity = mean_absolute_error(y_test.iloc[:, 1], y_pred[:, 1])
mae_pressure = mean_absolute_error(y_test.iloc[:, 2], y_pred[:, 2])

print(f"Mean Absolute Error (Temperature): {mae_temp}")
print(f"Mean Absolute Error (Humidity): {mae_humidity}")
print(f"Mean Absolute Error (Pressure): {mae_pressure}")


# In[18]:


y_test_1d = np.ravel(y_test) 
y_pred_1d = np.ravel(y_pred)

data = pd.DataFrame({'Actual': y_test_1d, 'Predicted': y_pred_1d})

sns.regplot(x='Actual', y='Predicted', data=data, scatter_kws={'color': 'blue'}, line_kws={'color': 'red', 'linestyle': '--'})
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted Temperatures")
plt.show()


# In[ ]:




