import pandas as pd
from process.data_merge_process.data_process.tools import *

'''
处理质量站的数据
将网格数据距离最近的质量站数据进行经纬度合并替换

'''

def __search_aq_grid_dict(aq_grid_dict,aq_longitude,aq_latitude):
    for aq_name_grid, gridTuple in aq_grid_dict.items():
        target_longitude = gridTuple[0]
        target_latitude = gridTuple[1]
        if aq_longitude == target_longitude and aq_latitude == target_latitude:
            return True
    return False

def ProcessAqData():
    aq_grid_dict = {}
    aq_station_data = pd.read_csv("./data/Beijing_AirQuality_Stations_cn_data.csv")
    aq_grid_data = pd.read_csv("./data/process_data/grid_18_meo.csv")
    df_aq_grid = pd.DataFrame(aq_grid_data,index=None,columns=['stationName','longitude','latitude','time','temperature','pressure','humidity','wind_direction','wind_speed'])
    df_aq_grid.rename(columns={'time':'utc_time','wind_speed':'wind_speed/kph'}, inplace = True)

    df_aq_grid_drop = df_aq_grid.drop_duplicates(['longitude','latitude'])
    for aq_index in aq_station_data.index:
        aq_name = aq_station_data.loc[aq_index].values[0]
        aq_longitude = aq_station_data.loc[aq_index].values[1]
        aq_latitude = aq_station_data.loc[aq_index].values[2]

        cur_min_dis = None
        cur_min_tuple = None
        for index in df_aq_grid_drop.index:
            longitude = df_aq_grid_drop.loc[index].values[1]
            latitude = df_aq_grid_drop.loc[index].values[2]
            dis = Distance1(latitude,longitude,aq_latitude,aq_longitude)
            if cur_min_dis is None:
                cur_min_dis = dis
                cur_min_tuple = (longitude, latitude)
            else:
                if dis <= cur_min_dis:
                    cur_min_dis = dis
                    cur_min_tuple = (longitude,latitude)

        aq_grid_dict[aq_name] = cur_min_tuple

    contact_result_grid = []
    for k,v in aq_grid_dict.items():
        aq_name = k
        aq_longitude = v[0]
        aq_latitude  = v[1]
        df_aq_grid_temp = df_aq_grid[(df_aq_grid.longitude == aq_longitude) & (df_aq_grid.latitude == aq_latitude)]
        df_aq_grid_temp.stationName[(df_aq_grid.longitude == aq_longitude) & (df_aq_grid.latitude == aq_latitude)] = aq_name
        contact_result_grid.append(df_aq_grid_temp)

    result_data = pd.concat(contact_result_grid)

    print('result_data:',result_data)
    result_data.to_csv('./data/process_data/merge_grid_meo.csv',index=False)

#ProcessAqData()