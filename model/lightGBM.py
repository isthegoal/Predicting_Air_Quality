#-*-coding:utf-8-*-


import pandas as pd
import lightgbm as lgb
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
import h5py
import  pickle
from pandas import DataFrame as DF

def smape_error(preds, train_data):
    labels = train_data.get_label()
    return 'error', np.mean(np.fabs(preds - labels) / (preds + labels) * 2), False
# 指标2 RMSE
def rmse(y_test, y):
    return np.sqrt(np.mean((y_test - y) ** 2))

def R2(y_test, y_true):
    return 1 - ((y_true - y_test) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()

# 自己跟据公式仿写Bias的计算函数
def Bias(y_true, y_test):
    return np.mean((y_true - y_test))

def f1(x):
    return np.log(x+1)
def rf1(x):
    return np.exp(x)-1

def get_score(pred, valid_y_exp):
    return np.mean(np.abs(pred - valid_y_exp) / (pred + valid_y_exp) * 2)

def get_pic(model, feature_name):
    ans = DF()
    print('is  is  ',len(feature_name))
    print('is  ',len(model.feature_importances_))
    ans['name'] = feature_name
    ans['score'] = model.feature_importances_
    #     print(ans[ans['score']>0].shape)
    print('获得最重要的名称')
    return ans.sort_values(by=['score'], ascending=False).reset_index(drop=True)



def pre_train_lgb_model(train_air='pm25'):
    '''
    使用这个函数可以训练三个模型（分别按指标）
    :param train_air:  指标名
    :return:
    '''
    f = h5py.File('./data/slice_eve_data/the_last_all__7_13_pca_com_800.h5', 'r')
    #f['data'].value存放的是时间戳 上空间的流量数据
    data = f['data'].value
    weidu_size=data.shape[1]

    pm25_data = pd.DataFrame(data,columns=[str(i) for i in range(0,weidu_size)])
    print(pm25_data.columns)
    #pm25_data=pm25_data.ix[1:400000,:]
    # print(pm25_data['1762'])
    # print(f1(pm25_data['1762']))

    #train_X, test_X, train_Y, test_Y = train_test_split(pm25_data[[str(i) for i in range(0,1762)]], f1(pm25_data['1762']), test_size=0.2, random_state=11)
    pm25_data = pm25_data.ix[1:600000, :]

    for nums in range(300,2000,50):
        pm25_data = pm25_data.ix[1:600000, :]
        train=pm25_data[[str(i) for i in range(0,weidu_size-1)]]
        y=pm25_data[str(weidu_size-1)]

        print('nums:', nums)
        #
        lgb_model = lgb.LGBMRegressor()
        # cv_model = cv(lgb_model, train_data[feature_name], train_label,  cv=10, scoring='f1')
        lgb_model.fit(train, y)

        print('小训练结束')
        feature_name1 = [str(i) for i in range(0, weidu_size - 1)]
        # print('用到的特征：：：',feature_name.ix[:nums])
        important_column = list(set(get_pic(lgb_model, feature_name1).head(nums)['name']))

        lgb_model = lgb.LGBMRegressor()
        # # cv_model = cv(lgb_model, train_data[feature_name], train_label,  cv=10, scoring='f1')
        lgb_model.fit(train[important_column], y)

        f = h5py.File(
            './data/slice_eve_data/the_last_all__7_13_pca_com_800.h5',
            'r')
        data = f['data'].value
        pm25_data = pd.DataFrame(data, columns=[str(i) for i in range(0, weidu_size)])


        # 打印所有数据集的时刻上的预测的分
        print('重要特恒长度为：',len(important_column))
        the_sum_score = 0
        the_sum_num = 0
        for i in range(750, 1200):
            start_num = 840 * i
            end_num = (i + 1) * 840

            print('start_num:', start_num)
            print('end_num:', end_num)

            pm_data = pm25_data.ix[int(start_num):int(end_num), :]

            pred_PM25 = lgb_model.predict(pm_data[important_column])

            score = get_score(pred_PM25, pm_data[str(weidu_size - 1)])
            print(str(i) + '次，计算留出集合上损失得分：', score)
            the_sum_score = the_sum_score + score
            the_sum_num = the_sum_num + 1
        # f['data'].value存放的是时间戳 上空间的流量数据
        print('探索特征数两 平均得分：', the_sum_score / the_sum_num)

        f = open('./record/lgb_model_record_1.txt', 'a+')
        f.write(str(nums) + '：' + str(the_sum_score / the_sum_num) + '\n\n')
        f.close()
        print('重要的列有：',important_column)
        f = h5py.File('./record/imp_lgb_feature_.h5', 'w')
        f['data']=[i.encode('utf-8') for i in important_column]
        f.close()



def model_lgb_train():
    print('---------       数据等信息的融入（对400个特征应该单独抽取出来）        -----------')
    f = h5py.File(
        './data/slice_eve_data/the_last_all__7_13_pca_com_800.h5',
        'r')
    # f['data'].value存放的是时间戳 上空间的流量数据
    data = f['data'].value
    weidu_size = data.shape[1]
    print('..............', weidu_size)
    important_column = ['135', '745', '227', '422', '607', '1297', '873', '1675', '1836', '302', '2377', '1489', '1203',
                        '1488', '1663', '662', '1738', '2529', '1648', '1215', '1332', '719', '348', '1683', '831',
                        '1094', '584', '2555', '961', '1333', '277', '95', '606', '1551', '1711', '1536', '4', '645',
                        '652', '2560', '1735', '448', '657', '2531', '554', '1440', '2553', '722', '699', '170', '687',
                        '1533', '1310', '1731', '468', '2151', '843', '1028', '1149', '1770', '1719', '324', '158',
                        '1426', '1772', '767', '120', '784', '1021', '369', '444', '0', '1430', '1647', '681', '60',
                        '48', '2538', '483', '1725', '314', '1548', '743', '1739', '697', '2528', '1749', '617', '518',
                        '1750', '1707', '1459', '1746', '1763', '710', '793', '698', '2532', '2547', '1743', '1402',
                        '2541', '254', '182', '1238', '1729', '122', '853', '2535', '290', '812', '410', '456', '2550',
                        '1758', '2540', '432', '278', '1101', '264', '206', '566', '1537', '694', '375', '2551', '381',
                        '1730', '691', '338', '783', '1394', '633', '602', '1736', '605', '592', '1784', '528', '2558',
                        '674', '376', '993', '1748', '885', '1365', '650', '360', '1808', '1377', '2548', '2554',
                        '2559', '33', '105', '1119', '1734', '1113', '532', '390', '433', '1578', '2543', '671', '1250',
                        '2', '1575', '9', '2556', '1490', '374', '160', '1653', '756', '1756', '768', '1381', '134',
                        '151', '350', '1718', '981', '1737', '470', '2544', '1671', '1716', '339', '37', '1742', '685',
                        '849', '724', '265', '1077', '39', '1744', '996', '372', '686', '1366', '530', '1713', '688',
                        '520', '604', '1755', '1768', '988', '1359', '1485', '805', '664', '1372', '1720', '1225',
                        '1323', '1751', '1623', '1406', '969', '1741', '319', '1173', '1672', '1154', '420', '1346',
                        '1572', '2536', '817', '1703', '2533', '1728', '2546', '700', '1200', '1767', '482', '1463',
                        '535', '1089', '1096', '291', '723', '758', '408', '1251', '1745', '1012', '507', '2561',
                        '1202', '384', '279', '2095', '262', '1629', '1775', '112', '1416', '820', '583', '1354',
                        '2262', '51', '387', '1753', '537', '1466', '1423', '385', '1322', '1660', '1353', '1635', '1',
                        '1689', '2545', '542', '1393', '1659', '1593', '2326', '1632', '2530', '1680', '1390', '231',
                        '1565', '489', '1526', '717', '590', '1286', '1053', '1726', '1721', '84', '24', '727', '1378',
                        '146', '693', '1701', '1754', '781', '1315', '733', '1065', '1779', '517', '97', '1335', '2539',
                        '446', '1599', '945', '502', '218', '1418', '1237', '1515', '531', '1137', '1550', '362', '38',
                        '242', '2184', '962', '861', '2197', '513', '1705', '1776', '458', '1041', '2542', '1727',
                        '1563', '237', '1502', '1104', '948', '601', '775', '1221', '15', '1677', '230', '386', '2386',
                        '2537', '2413', '1262', '1587', '266', '611', '87', '795', '1330', '916', '2534', '1083', '434',
                        '734', '631', '585', '1477', '1668', '1699', '669', '195', '1740', '2549', '579', '626', '637',
                        '1611', '1717', '1589', '1695', '1298', '2552', '928', '1806', '1723', '3', '643', '36', '506',
                        '1684', '984', '1058', '1760', '649', '72', '2557', '398']
    pm25_data = pd.DataFrame(data, columns=[str(i) for i in range(0, weidu_size)])

    print('---------       模型的训练        -----------')
    pm25_data_600 = pm25_data.ix[1:600000, :]
    train = pm25_data_600[[str(i) for i in range(0, weidu_size - 1)]]
    y = pm25_data_600[str(weidu_size - 1)]
    # 使用fit方式训练可以最大的利用所有的数据集没有预留0.2

    lgb_model = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=1000, colsample_bytree=0.9, max_depth=10,
                                  num_leaves=200, min_data_in_leaf=20, reg_alpha=0.001, feature_fraction=1,
                                  bagging_fraction=0.9)
    # # cv_model = cv(lgb_model, train_data[feature_name], train_label,  cv=10, scoring='f1')
    lgb_model.fit(train[important_column], y)

    the_sum_score = 0
    the_sum_num = 0
    for i in range(750, 1200):
        start_num = 840 * i
        end_num = (i + 1) * 840

        print('start_num:', start_num)
        print('end_num:', end_num)

        pm_data = pm25_data.ix[int(start_num):int(end_num), :]

        pred_PM25 = lgb_model.predict(pm_data[important_column])

        score = get_score(pred_PM25, pm_data[str(weidu_size - 1)])
        print(str(i) + '次，计算留出集合上损失得分：', score)
        the_sum_score = the_sum_score + score
        the_sum_num = the_sum_num + 1
    # f['data'].value存放的是时间戳 上空间的流量数据
    print('探索特征数两 平均得分：', the_sum_score / the_sum_num)

    from sklearn.externals import joblib
    joblib.dump(lgb_model, './data/save_model/lgb_best_yishen_0.4184.pkl')

def model_lgb_predict():

    print('---------       数据等信息的融入（对400个特征应该单独抽取出来）        -----------')
    f = h5py.File(
        './data/slice_eve_data/the_last_all__7_13_pca_com_800.h5',
        'r')
    # f['data'].value存放的是时间戳 上空间的流量数据
    data = f['data'].value
    weidu_size = data.shape[1]
    print('..............', weidu_size)
    model_path = '../data/save_model/lgb_best_yishen_0.4184.model'
    lgb_model = pickle.load(open(model_path, 'rb'))

    important_column = ['135', '745', '227', '422', '607', '1297', '873', '1675', '1836', '302', '2377', '1489', '1203',
                        '1488', '1663', '662', '1738', '2529', '1648', '1215', '1332', '719', '348', '1683', '831',
                        '1094', '584', '2555', '961', '1333', '277', '95', '606', '1551', '1711', '1536', '4', '645',
                        '652', '2560', '1735', '448', '657', '2531', '554', '1440', '2553', '722', '699', '170', '687',
                        '1533', '1310', '1731', '468', '2151', '843', '1028', '1149', '1770', '1719', '324', '158',
                        '1426', '1772', '767', '120', '784', '1021', '369', '444', '0', '1430', '1647', '681', '60',
                        '48', '2538', '483', '1725', '314', '1548', '743', '1739', '697', '2528', '1749', '617', '518',
                        '1750', '1707', '1459', '1746', '1763', '710', '793', '698', '2532', '2547', '1743', '1402',
                        '2541', '254', '182', '1238', '1729', '122', '853', '2535', '290', '812', '410', '456', '2550',
                        '1758', '2540', '432', '278', '1101', '264', '206', '566', '1537', '694', '375', '2551', '381',
                        '1730', '691', '338', '783', '1394', '633', '602', '1736', '605', '592', '1784', '528', '2558',
                        '674', '376', '993', '1748', '885', '1365', '650', '360', '1808', '1377', '2548', '2554',
                        '2559', '33', '105', '1119', '1734', '1113', '532', '390', '433', '1578', '2543', '671', '1250',
                        '2', '1575', '9', '2556', '1490', '374', '160', '1653', '756', '1756', '768', '1381', '134',
                        '151', '350', '1718', '981', '1737', '470', '2544', '1671', '1716', '339', '37', '1742', '685',
                        '849', '724', '265', '1077', '39', '1744', '996', '372', '686', '1366', '530', '1713', '688',
                        '520', '604', '1755', '1768', '988', '1359', '1485', '805', '664', '1372', '1720', '1225',
                        '1323', '1751', '1623', '1406', '969', '1741', '319', '1173', '1672', '1154', '420', '1346',
                        '1572', '2536', '817', '1703', '2533', '1728', '2546', '700', '1200', '1767', '482', '1463',
                        '535', '1089', '1096', '291', '723', '758', '408', '1251', '1745', '1012', '507', '2561',
                        '1202', '384', '279', '2095', '262', '1629', '1775', '112', '1416', '820', '583', '1354',
                        '2262', '51', '387', '1753', '537', '1466', '1423', '385', '1322', '1660', '1353', '1635', '1',
                        '1689', '2545', '542', '1393', '1659', '1593', '2326', '1632', '2530', '1680', '1390', '231',
                        '1565', '489', '1526', '717', '590', '1286', '1053', '1726', '1721', '84', '24', '727', '1378',
                        '146', '693', '1701', '1754', '781', '1315', '733', '1065', '1779', '517', '97', '1335', '2539',
                        '446', '1599', '945', '502', '218', '1418', '1237', '1515', '531', '1137', '1550', '362', '38',
                        '242', '2184', '962', '861', '2197', '513', '1705', '1776', '458', '1041', '2542', '1727',
                        '1563', '237', '1502', '1104', '948', '601', '775', '1221', '15', '1677', '230', '386', '2386',
                        '2537', '2413', '1262', '1587', '266', '611', '87', '795', '1330', '916', '2534', '1083', '434',
                        '734', '631', '585', '1477', '1668', '1699', '669', '195', '1740', '2549', '579', '626', '637',
                        '1611', '1717', '1589', '1695', '1298', '2552', '928', '1806', '1723', '3', '643', '36', '506',
                        '1684', '984', '1058', '1760', '649', '72', '2557', '398']
    pm25_data = pd.DataFrame(data, columns=[str(i) for i in range(0, weidu_size)])

    print('---------       模型的训练        -----------')
    pm25_data_600 = pm25_data.ix[1:600000, :]
    train = pm25_data_600[[str(i) for i in range(0, weidu_size - 1)]]
    y = pm25_data_600[str(weidu_size - 1)]
    # 使用fit方式训练可以最大的利用所有的数据集没有预留0.2
    the_sum_score = 0
    the_sum_num = 0

    the_rmse_score = 0
    the_r2_score = 0
    the_mae_score = 0
    the_bias_score = 0

    for i in range(750, 1200):
        start_num = 840 * i
        end_num = (i + 1) * 840

        print('start_num:', start_num)
        print('end_num:', end_num)

        pm_data = pm25_data.ix[int(start_num):int(end_num), :]

        pred_PM25 = lgb_model.predict(pm_data[important_column])

        # 分别计算   rmse_score  r2_score    mae_score    bias_score
        score = get_score(pred_PM25, pm_data[str(weidu_size - 1)])

        rmse_score = rmse(pred_PM25, pm_data[str(weidu_size - 1)])
        r2_score = R2(pred_PM25, pm_data[str(weidu_size - 1)])

        mae_score = metrics.mean_absolute_error(pred_PM25, pm_data[str(weidu_size - 1)])
        bias_score = Bias(pred_PM25, pm_data[str(weidu_size - 1)])

        print(str(i) + '次，计算留出集合上损失得分：', score)
        the_sum_score = the_sum_score + score
        the_rmse_score = the_rmse_score + rmse_score
        the_r2_score = the_r2_score + r2_score
        the_mae_score = the_mae_score + mae_score
        the_bias_score = the_bias_score + bias_score

        the_sum_num = the_sum_num + 1
    # f['data'].value存放的是时间戳 上空间的流量数据
    print('探索特征数两 平均得分：', the_sum_score / the_sum_num)
    print('rmse 平均得分：', the_rmse_score / the_sum_num)
    print('r2 平均得分：', the_r2_score / the_sum_num)
    print('mae 平均得分：', the_mae_score / the_sum_num)
    print('bia 平均得分：', the_bias_score / the_sum_num)

    score_recore_list = []
    for she_end_hour in range(0, 24):
        for i in range(800, 801):
            start_num = 840 * 750 + she_end_hour
            end_num = (1202) * 840
            # end_num=start_num+

            print('start_num:', start_num)
            print('end_num:', end_num)

            # pm_data = pm25_data.ix[int(start_num):int(end_num), :

            pm_data = pm25_data.ix[[i for i in range(start_num, end_num, 24)], :]

            pred_PM25 = lgb_model.predict(pm_data[important_column])

            # 分别计算   rmse_score  r2_score    mae_score    bias_score
            score = get_score(pred_PM25, pm_data[str(weidu_size - 1)])

            print(str(i) + '次，计算留出集合上损失得分：', score)
            score_recore_list.append(score)

    print('1-3 小时上平均smape指标值：',np.mean(score_recore_list[0:3]))
    print('4-9 小时上平均smape指标值：',np.mean(score_recore_list[3:9]))
    print('9-16 小时上平均smape指标值：',np.mean(score_recore_list[9:16]))
    print('17-24 小时上平均smape指标值：',np.mean(score_recore_list[16:24]))
