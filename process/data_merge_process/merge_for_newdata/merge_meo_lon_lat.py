import pandas as pd
#用于为每个网格获取经纬度(step1)
def get_meo_lon_lat():
    meo_18=pd.read_csv("./data/origion_data/18weather_2018_3_31.csv",low_memory=False)
    meo_17=pd.read_csv("./data/beijing_17_18_meo.csv")
    lon_lat=pd.DataFrame(meo_17,columns=['station_id','longitude','latitude'])
    dr_lon_lat=lon_lat.drop_duplicates(['longitude', 'latitude'])
    df_meo_18=pd.DataFrame(meo_18)
    meo=pd.merge( df_meo_18,dr_lon_lat,on ='station_id',how='inner')
    meo.to_csv('./data/process_data/18_meo_station.csv',
                       index=False)
    print(meo.head())


#get_meo_lon_lat()