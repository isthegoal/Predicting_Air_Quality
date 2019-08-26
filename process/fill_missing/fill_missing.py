import pandas as pd
import datetime
import numpy as np

pd.set_option('display.max_columns', 5000)

#检查是否有缺失时间点
def see_missing_date():
    air_meo = pd.read_csv(
        './data/process_data/final_merge_aq_grid_meo_deal_weather.csv',
        parse_dates=['utc_time'])
    weather_predict = pd.read_csv(
        './data/process_data/weather_prediction.csv',
        parse_dates=['datetime'])

    # 限制气象信息的时刻点
    start_date = pd.to_datetime('2018-03-31 07:00:00')
    end_date = pd.to_datetime('2018-05-29 06:00:00')
    air_meo = air_meo[
        (start_date <= air_meo['utc_time']) & (air_meo['utc_time'] <= end_date)]


    print(air_meo['utc_time'].shape)

    print(str(air_meo['utc_time']))
    #################################  输出  气象信息和 天气预报各缺少那些时间点   ########################
    start_date = pd.to_datetime('2018-03-31 07:00:00')
    end_date = pd.to_datetime('2018-05-29 06:00:00')
    air_mo_loss_time = []
    utc_time_list = str(list(air_meo['utc_time']))
    while (start_date <= end_date):
        if str(start_date) not in utc_time_list:
            air_mo_loss_time.append(start_date)
        start_date = start_date + datetime.timedelta(hours=1)

    start_date = pd.to_datetime('2018-03-31 07:00:00')
    end_date = pd.to_datetime('2018-05-29 06:00:00')
    weather_predict_loss_time = []
    datetime_list = str(list(weather_predict['datetime']))
    while (start_date <= end_date):
        if str(start_date) not in datetime_list:
            weather_predict_loss_time.append(start_date)
        start_date = start_date + datetime.timedelta(hours=1)

    print('气象数据缺失的时刻点有：', air_mo_loss_time)
    print('天气预报数据缺失的时刻点有：', weather_predict_loss_time)

def missing_fill_air_pul():
    '''

    插时间点的思路是，先找缺失点开始的前多少个时间点是存在的，记录向前间隔为i,  之后找缺失点开始的后多少个时间点是存在的，记录向前间隔为j
    这样i+j即为连续缺失的时间点数量，如果连续缺5个就不作处理了。


    '''

    air_meo = pd.read_csv(
        './data/process_data/final_merge_aq_grid_meo_deal_weather.csv',
        parse_dates=['utc_time'])
    weather_predict = pd.read_csv(
        './data/process_data/weather_prediction.csv',
        parse_dates=['datetime'])

    # 限制气象信息的时刻点
    start_date = pd.to_datetime('2018-03-31 07:00:00')
    end_date = pd.to_datetime('2018-05-29 06:00:00')
    air_meo = air_meo[
        (start_date <= air_meo['utc_time']) & (air_meo['utc_time'] <= end_date)]

    print('标记1',air_meo.describe())
    data = air_meo
    station_name = pd.read_csv(
        './data/Beijing_AirQuality_Stations_cn_data.csv')
    station_name = pd.DataFrame(station_name, columns=['stationId'])
    print(station_name)
    # 去除重复日期
    train_data = data
    data = pd.DataFrame(data)
    print(data.info())
    # 去除重复的日期
    data.drop_duplicates(['utc_time'], inplace=True)
    print("原数据时间戳的数量：", data.iloc[:, 0].size)
    # data.to_csv('/home/fly/PycharmProjects/gmh_kdd/DeepST-KDD for_train/ma_util/data_process/data_outputs/test_fill_5_hours.csv')
    # data=pd.read_csv('/home/fly/PycharmProjects/gmh_kdd/DeepST-KDD for_train/ma_util/data_process/data_outputs/test_fill_5_hours.csv')
    # print(data.info())
    # print(data.head())
    data.set_index("utc_time", inplace=True)
    # print (data.index)
    min_time = data.index.min()
    max_time = data.index.max()
    print("最小时间点：", min_time)
    print("最大时间点：", max_time)
    min_time = datetime.datetime.strptime(str(min_time), '%Y-%m-%d %H:%M:%S')
    max_time = datetime.datetime.strptime(str(max_time), '%Y-%m-%d %H:%M:%S')
    delta_all = max_time - min_time
    hours_should = delta_all.total_seconds() / 3600 + 1
    print("应该有的时间数：", hours_should)
    # ---------------------------------------------整个小时，缺失的时间点----------------------------
    delta = datetime.timedelta(hours=1)
    # --------------------------------------------missing_hours_str[]  存储缺失的时间点
    missing_hours_str = []
    time = min_time
    missing_hours = []
    while time <= max_time:
        if datetime.date.strftime(time, '%Y-%m-%d %H:%M:%S') not in data.index:
            missing_hours.append(time)
            missing_hours_str.append(datetime.date.strftime(time, '%Y-%m-%d %H:%M:%S'))
        time += delta
    print("缺失时间绰的数量：", len(missing_hours_str))
    # missing_hours_str=pd.DataFrame(missing_hours_str)
    # missing_hours_str.to_csv('/home/fly/PycharmProjects/gmh_kdd/DeepST-KDD for_train/ma_util/data_process/data_outputs/test_absent_time.csv')
    keep_hours = []
    drop_hours = []
    # --------------------------------------------先对小于5小时的进行填充
    delta = datetime.timedelta(hours=1)
    for hour in missing_hours_str:
        # print(hour)
        time = datetime.datetime.strptime(hour, '%Y-%m-%d %H:%M:%S')
        # 前边第几个是非空的
        found_for = False
        i = 0
        while not found_for:
            i += 1
            for_time = time - i * delta
            for_time_str = datetime.date.strftime(for_time, '%Y-%m-%d %H:%M:%S')
            if for_time_str in data.index:
                for_row = data.loc[for_time_str]
                # for_row=pd.DataFrame(for_row)
                # for_row=pd.DataFrame(for_row,columns=['temperature','pressure','humidity','wind_direction','wind_speed/kph','weather','NO2','CO','SO2','PM2.5','PM10','O3'])
                # for_row=for_row['temperature','pressure','humidity','wind_direction','wind_speed/kph','weather','NO2','CO','SO2','PM2.5','PM10','O3']
                utc_time = for_row.name
                # 找到的前面的要插入的某一时刻35个站点的数据
                insert_for = train_data.loc[train_data['utc_time'] == utc_time]
                print("insert_for:\n")

                insert_for = pd.DataFrame(insert_for[['temperature', 'pressure', 'humidity', 'wind_direction',
                                                      'wind_speed/kph', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3']])
                # insert_for= pd.DataFrame(insert_for,columns=['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3'])
                print(insert_for)
                # print(insert_for)
                for_step = i
                found_for = True
        # 后边第几个是非空的
        found_back = False
        j = 0
        while not found_back:
            j += 1
            back_time = time + j * delta
            back_time_str = datetime.date.strftime(back_time, '%Y-%m-%d %H:%M:%S')
            if back_time_str in data.index:
                back_row = data.loc[back_time_str]
                utc_time = back_row.name
                insert_back = train_data.loc[train_data['utc_time'] == utc_time]
                weather = insert_back['weather']
                print('weather:', weather)
                # print(insert_back)
                #  print("insert_back:\n")
                insert_back = pd.DataFrame(insert_back[['temperature', 'pressure', 'humidity', 'wind_direction',
                                                        'wind_speed/kph', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3']])

                back_step = j
                found_back = True
        # print(for_step, back_step)
        all_steps = for_step + back_step
        if all_steps > 5:
            drop_hours.append(hour)
        else:
            keep_hours.append(hour)
            # 插值
            # insert_for与inset——back不包括站点数据，因为站点数据属于字符串，不能做运算，以及weather也不能做运算,因为weather是离散型的，所以需要再拼接
            insert_for = np.array(insert_for)
            insert_back = np.array(insert_back)
            delata_values = insert_for - insert_back

            # 建立要插入的某一时刻的pandas，目前只有固定的时间与35 个站点两列
            df_inset = pd.DataFrame(columns=['utc_time', 'stationId_aq', 'weather'])
            df_inset['stationId_aq'] = station_name['stationId']
            df_inset['utc_time'] = hour
            df_inset['weather'] = weather
            # print(df_inset)
            df_insert_other_colunms = pd.DataFrame((insert_for + (for_step / all_steps) * delata_values),
                                                   columns=['temperature', 'pressure', 'humidity', 'wind_direction',
                                                            'wind_speed/kph', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10',
                                                            'O3'])
            # 将时间戳，站点以及对应的co等属性值进行拼接
            df_inset = pd.concat([df_inset, df_insert_other_colunms], axis=1)
            # 将要插入的信息 拼接到原数据缺失时间戳中

            print("插入的某一时刻35个站点的信息;\n")
            print("hour:\n", hour)
            print(df_inset)
            train_data = pd.concat([train_data, df_inset], axis=0)
            train_data = train_data[
                ['utc_time', 'stationId_aq', 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph',
                 'weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3']]

    print("小于5个小时的可以填充的数量:", len(keep_hours))

    print('标记2',train_data.describe())
    train_data.to_csv(
        './data/process_data/fill_5_hour_data.csv')


def fill_missing_data_xiangxi():
    '''
    在时间点齐全的情况下，如果丢失站点数据，这里进行进行补全
    好吧，最后line:    49560
    空气质量数据长度： (49576, 16)
    天气预报数据长度： (49560, 70)
    一个站点的都没少，发现是空气质量处理的结果还多了16个，看样子是没有最后去重的原因
    '''
    air_data = pd.read_csv(
        './data/feature_eng_data/air_data.csv',
        parse_dates=['utc_time'])
    weather_pred_data = pd.read_csv(
        './data/feature_eng_data/weather_pred.csv',
        parse_dates=['datetime'])

    air_min_date = np.min(air_data['utc_time'])
    air_max_date = np.max(air_data['utc_time'])
    weather_pred_min_date = np.min(weather_pred_data['datetime'])
    weather_pred_max_date = np.max(weather_pred_data['datetime'])
    print('空气质量数据长度：', air_data.shape)
    print('天气预报数据长度：', weather_pred_data.shape)
    print(air_min_date)
    print(air_max_date)
    print(weather_pred_min_date)
    print(weather_pred_max_date)

    print(set(air_data['stationId_aq']))
    start_date = datetime.datetime.strptime(str('2018-03-31 07:00:00'), '%Y-%m-%d %H:%M:%S')
    max_time = datetime.datetime.strptime(str('2018-05-29 06:00:00'), '%Y-%m-%d %H:%M:%S')

    linenum = 0
    # 对于历史气象数据，去判断缺失了哪些站点数据
    for station in ['donggaocun_aq', 'nansanhuan_aq', 'nongzhanguan_aq', 'pingchang_aq', 'yungang_aq',
                    'wanshouxigong_aq',
                    'aotizhongxin_aq', 'huairou_aq', 'miyunshuiku_aq', 'tongzhou_aq', 'dingling_aq', 'fangshan_aq',
                    'pinggu_aq',
                    'dongsi_aq', 'liulihe_aq', 'xizhimenbei_aq', 'badaling_aq', 'shunyi_aq', 'wanliu_aq',
                    'zhiwuyuan_aq', 'guanyuan_aq',
                    'tiantan_aq', 'yanqin_aq', 'daxing_aq', 'fengtaihuayuan_aq', 'mentougou_aq', 'yufa_aq',
                    'beibuxinqu_aq', 'dongsihuan_aq',
                    'gucheng_aq', 'qianmen_aq', 'miyun_aq', 'yizhuang_aq', 'yongledian_aq', 'yongdingmennei_aq']:
        start_date = datetime.datetime.strptime(str('2018-03-31 07:00:00'), '%Y-%m-%d %H:%M:%S')
        while start_date <= max_time:
            #
            # 判断这个条件是否存在表格里面
            need_line = air_data[(air_data['utc_time'] == start_date) & (air_data['stationId_aq'] == station)]
            if need_line.empty == True:
                print('有位置缺了')
            start_date = start_date + datetime.timedelta(hours=1)
            linenum += 1

    print('line:   ', linenum)


def check_data_null():
    '''
    检查数据中的空值，在进行合并之前进行检查，不然会出事的。  有空值不会让训练的。   发现在windBearing   windSpeed  两列上是有缺失值的。[线性插值处理好的]

    所以得对这进行缺失值处理。

    之后空气质量中weather是有缺失数值的
    :return:
    '''
    air_data = pd.read_csv(
        './data/process_data/fill_5_hour_data.csv',
        parse_dates=['utc_time'])
    weather_pred_data = pd.read_csv(
        './data/feature_eng_data/weather_pred.csv',
        parse_dates=['datetime'])
    ######################### 检查天气预报  ###########################
    weather_pred_data = weather_pred_data.drop('datetime', axis=1)
    print('air:', weather_pred_data.iloc[:30, :])

    weather_pred_data = weather_pred_data.interpolate()

    is_nan_air = np.isnan(weather_pred_data).any()
    print(is_nan_air)
    # print('isnan:',is_nan_air['dewPoint','humidity','temperature'].isnull())
    for index, i in enumerate(list(is_nan_air)):
        if (i == True):
            print('::::', index)
    ######################### 检查空气质量  ###########################
    air_data_U = air_data['Unnamed: 0']
    air_data = air_data.drop('Unnamed: 0', axis=1)
    air_data_time = air_data['utc_time']
    air_data = air_data.drop('utc_time', axis=1)
    air_data_sta = air_data['stationId_aq']
    air_data = air_data.drop('stationId_aq', axis=1)
    # air_data = air_data.drop('Unnamed: 0', axis=1)
    air_data = air_data.interpolate()
    # air_data['Unnamed: 0']=air_data_U
    air_data['utc_time'] = air_data_time
    air_data['stationId_aq'] = air_data_sta
    air_data.to_csv(
        './data/feature_eng_data/air_data_fill_wea.csv')
