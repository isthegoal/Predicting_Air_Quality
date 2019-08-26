import pandas as pd
#用于为每个网格获取经纬度(step1)
def get_grid_lon_lat():
    grid_18=pd.read_csv("./data/origion_data/gird_2018_3_31.csv",low_memory=False)
    grid_17=pd.read_csv("./data/Beijing_historical_meo_grid.csv",dtype={'stationName':str})
    lon_lat=pd.DataFrame(grid_17,columns=['stationName','longitude','latitude'])
    #此步骤很重要，否则会产生memory错误
    dr_lon_lat=lon_lat.drop_duplicates(['longitude', 'latitude'])
    df_grid_18=pd.DataFrame(grid_18)
    meo=pd.merge( df_grid_18,dr_lon_lat,on ='stationName',how='inner')
    meo.to_csv('./data/process_data/grid_18_meo.csv',
                       index=False)
    print(meo.head())
#get_grid_lon_lat()