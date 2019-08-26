import pandas as pd

def trans_weather_to_num():
    data = pd.read_csv('./data/process_data/final_merge_aq_grid_meo_deal_new.csv')
    print(data.info())
    #去除所有weather为空的数据，～表示去除的一种方法，因为连续20天没有数据
    #data = data[~data['weather'].isnull()]
    data.loc[data["weather"] == "Sunny/clear", "weather"] = 1
    data.loc[data["weather"] == "Haze", "weather"] = 2
    data.loc[data["weather"] == "Snow", "weather"] = 3
    data.loc[data["weather"] == "Fog", "weather"] = 4
    data.loc[data["weather"] == "Rain", "weather"] = 5
    data.loc[data["weather"] == "Dust", "weather"] = 6
    data.loc[data["weather"] == "Sand", "weather"] = 7
    data.loc[data["weather"] == "Sleet", "weather"] = 8
    data.loc[data["weather"] == "Rain/Snow with Hail", "weather"] = 9
    data.loc[data["weather"] == "Rain with Hail", "weather"] = 10
    data.loc[data["weather"] == "Hail", "weather"] = 11
    data.loc[data["weather"] == "Cloudy", "weather"] = 12
    data.loc[data["weather"] == "Light Rain", "weather"] = 13
    data.loc[data["weather"] == "Thundershower", "weather"] = 14
    data.loc[data["weather"] == "Overcast", "weather"] = 15

    data.loc[data['weather'].isnull(),"weather"]=1
    print (data.head())
    print(data.weather)
    data = data.interpolate()
    data = data[['stationId_aq', 'utc_time', 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3']]
    data.to_csv('./data/process_data/final_merge_aq_grid_meo_deal_weather.csv')
    print(data.info())
    data1 = pd.read_csv('./data/process_data/final_merge_aq_grid_meo_deal_weather.csv')
    data_sort = pd.DataFrame(data1.sort_values(by = ['utc_time', 'stationId_aq']))
    data_sort = data_sort[['utc_time','stationId_aq','temperature','pressure','humidity','wind_direction','wind_speed/kph','weather','NO2','CO','SO2','PM2.5','PM10','O3']]
    data_sort.to_csv('./data/process_data/final_merge_aq_grid_meo_deal_weather.csv', index=False)
    data_sort.corr().to_csv('./data/process_data/feature_important_show.csv')

