#-*-coding:utf-8-*-
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import h5py
import  pickle
from xgboost import XGBRegressor as XGBR
from pandas import DataFrame as DF
from sklearn import metrics


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
#

def get_pic(model, feature_name):
    ans = DF()
    print('is  is  ',len(feature_name))
    print('is  ',len(model.feature_importances_))
    ans['name'] = feature_name
    ans['score'] = model.feature_importances_
    #     print(ans[ans['score']>0].shape)
    print('获得最重要的名称')
    return ans.sort_values(by=['score'], ascending=False).reset_index(drop=True)

def pre_model_xgb_train():
    f = h5py.File(
        './data/slice_eve_data/the_last_all__7_13_pca_com_800.h5',
        'r')
    # f['data'].value存放的是时间戳 上空间的流量数据
    data = f['data'].value
    weidu_size = data.shape[1]

    pm25_data = pd.DataFrame(data, columns=[str(i) for i in range(0, weidu_size)])
    pm25_data_60 = pm25_data.ix[1:600000, :]

    train = pm25_data_60[[str(i) for i in range(0, weidu_size - 1)]]
    y = pm25_data_60[str(weidu_size - 1)]
    print('小训练结束')

    reg = XGBR()
    reg.fit(train, y)

    feature_name1 = [str(i) for i in range(0, weidu_size - 1)]

    nums=300  #important feature number
    important_column = list(set(get_pic(reg, feature_name1).head(nums)['name']))

    reg1 = XGBR()
    reg1.fit(train[important_column], y)

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

        pred_PM25 = reg1.predict(pm_data[important_column])

        score = get_score(pred_PM25, pm_data[str(weidu_size - 1)])
        print(str(i) + '次，计算留出集合上损失得分：', score)
        the_sum_score = the_sum_score + score
        the_sum_num = the_sum_num + 1
    # f['data'].value存放的是时间戳 上空间的流量数据
    print('探索特征数两 平均得分：', the_sum_score / the_sum_num)
    print('重要的列有：',important_column)


def model_xgb_train():
    f = h5py.File(
        './data/slice_eve_data/the_last_all__7_13_pca_com_800.h5',
        'r')
    # f['data'].value存放的是时间戳 上空间的流量数据
    data = f['data'].value
    weidu_size = data.shape[1]

    pm25_data = pd.DataFrame(data, columns=[str(i) for i in range(0, weidu_size)])
    pm25_data_60 = pm25_data.ix[1:600000, :]

    train = pm25_data_60[[str(i) for i in range(0, weidu_size - 1)]]
    y = pm25_data_60[str(weidu_size - 1)]

    important_column = ['1322', '2184', '429', '1430', '2555', '768', '1473', '40', '336', '1739', '1734', '544', '24',
                        '1251', '1540', '1274', '194', '693', '1757', '705', '1173', '432', '793', '362', '195', '1394',
                        '279', '1284', '288', '458', '1429', '1549', '396', '669', '2542', '1505', '1772', '1642',
                        '753', '1076', '1281', '1106', '542', '1442', '2540', '1469', '1716', '681', '518', '1457',
                        '1269', '813', '1083', '1279', '158', '434', '1775', '871', '657', '1659', '1737', '691', '455',
                        '555', '1707', '1567', '264', '1684', '1552', '2559', '1414', '1651', '1203', '1464', '820',
                        '729', '1660', '1545', '99', '1438', '796', '1509', '2539', '384', '1730', '127', '1776',
                        '1701', '1729', '2528', '1626', '1493', '12', '650', '647', '482', '211', '1161', '1107',
                        '1675', '1311', '903', '314', '1736', '1623', '1197', '964', '645', '783', '494', '1299',
                        '1541', '537', '549', '2541', '782', '182', '916', '1725', '348', '1238', '2139', '381', '206',
                        '28', '146', '4', '444', '369', '2560', '2549', '1250', '1740', '1726', '890', '169', '1711',
                        '1752', '1722', '225', '1557', '554', '597', '1149', '614', '183', '1447', '1599', '697', '398',
                        '825', '177', '1342', '698', '734', '1239', '16', '1418', '372', '861', '877', '446', '147',
                        '204', '928', '1413', '2561', '506', '1488', '2530', '578', '674', '2041', '837', '2543',
                        '1320', '949', '468', '1395', '572', '1754', '302', '1226', '291', '1742', '148', '1131', '276',
                        '1717', '710', '1671', '70', '1735', '1354', '1779', '873', '1514', '945', '1440', '1310',
                        '626', '1435', '60', '1721', '2558', '2534', '590', '1663', '1491', '499', '1720', '1728',
                        '633', '360', '1767', '1763', '495', '1495', '1613', '110', '324', '1713', '207', '1081',
                        '1577', '218', '1648', '2531', '2546', '358', '2556', '746', '2', '988', '779', '2545', '1485',
                        '1209', '1478', '1176', '1468', '976', '638', '979', '1743', '2537', '170', '1647', '2552', '9',
                        '1718', '1012', '1334', '1738', '876', '662', '2535', '1683', '1687', '1060', '607', '1179',
                        '865', '1378', '2532', '2551', '108', '530', '1512', '770', '237', '1719', '849', '1759',
                        '1636', '721', '1477', '1831', '1589', '0', '2553', '2550', '2533', '1503', '386', '1125',
                        '1579', '1695', '453', '1543', '685', '1833', '1069', '351', '1741', '1611', '1591', '1497',
                        '1404', '819']
    reg1 = XGBR(learning_rate=0.05,
                n_estimators=600,
                max_depth=5,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0,
                reg_alpha=0,
                reg_lambda=1, scale_pos_weight=1, n_jobs=-1)

    # # cv_model = cv(lgb_model, train_data[feature_name], train_label,  cv=10, scoring='f1')
    reg1.fit(train[important_column], y)
    print('重要特恒长度为：', len(important_column))
    the_sum_score = 0
    the_sum_num = 0
    for i in range(750, 1200):
        start_num = 840 * i
        end_num = (i + 1) * 840
        print('start_num:', start_num)
        print('end_num:', end_num)
        pm_data = pm25_data.ix[int(start_num):int(end_num), :]

        pred_PM25 = reg1.predict(pm_data[important_column])

        score = get_score(pred_PM25, pm_data[str(weidu_size - 1)])
        print(str(i) + '次，计算留出集合上损失得分：', score)
        the_sum_score = the_sum_score + score
        the_sum_num = the_sum_num + 1
    # f['data'].value存放的是时间戳 上空间的流量数据
    print('探索特征数两 平均得分：', the_sum_score / the_sum_num)
    print('重要的列有：', important_column)

    # 模型存储
    model_file = './data/save_model/xgb_best_yishen_0.425.model'
    with open(model_file, 'wb') as fout:
        pickle.dump(reg1, fout)
    #
def model_xgb_predict():
    f = h5py.File(
        './data/slice_eve_data/the_last_all__7_13_pca_com_800.h5',
        'r')
    # f['data'].value存放的是时间戳 上空间的流量数据
    data = f['data'].value
    weidu_size = data.shape[1]
    pm25_data = pd.DataFrame(data, columns=[str(i) for i in range(0, weidu_size)])

    model_path = './data/save_model/xgb_best_yishen_0.425.model'
    xgb_model = pickle.load(open(model_path, 'rb'))

    important_column = ['1322', '2184', '429', '1430', '2555', '768', '1473', '40', '336', '1739', '1734', '544', '24',
                        '1251', '1540', '1274', '194', '693', '1757', '705', '1173', '432', '793', '362', '195', '1394',
                        '279', '1284', '288', '458', '1429', '1549', '396', '669', '2542', '1505', '1772', '1642',
                        '753', '1076', '1281', '1106', '542', '1442', '2540', '1469', '1716', '681', '518', '1457',
                        '1269', '813', '1083', '1279', '158', '434', '1775', '871', '657', '1659', '1737', '691', '455',
                        '555', '1707', '1567', '264', '1684', '1552', '2559', '1414', '1651', '1203', '1464', '820',
                        '729', '1660', '1545', '99', '1438', '796', '1509', '2539', '384', '1730', '127', '1776',
                        '1701', '1729', '2528', '1626', '1493', '12', '650', '647', '482', '211', '1161', '1107',
                        '1675', '1311', '903', '314', '1736', '1623', '1197', '964', '645', '783', '494', '1299',
                        '1541', '537', '549', '2541', '782', '182', '916', '1725', '348', '1238', '2139', '381', '206',
                        '28', '146', '4', '444', '369', '2560', '2549', '1250', '1740', '1726', '890', '169', '1711',
                        '1752', '1722', '225', '1557', '554', '597', '1149', '614', '183', '1447', '1599', '697', '398',
                        '825', '177', '1342', '698', '734', '1239', '16', '1418', '372', '861', '877', '446', '147',
                        '204', '928', '1413', '2561', '506', '1488', '2530', '578', '674', '2041', '837', '2543',
                        '1320', '949', '468', '1395', '572', '1754', '302', '1226', '291', '1742', '148', '1131', '276',
                        '1717', '710', '1671', '70', '1735', '1354', '1779', '873', '1514', '945', '1440', '1310',
                        '626', '1435', '60', '1721', '2558', '2534', '590', '1663', '1491', '499', '1720', '1728',
                        '633', '360', '1767', '1763', '495', '1495', '1613', '110', '324', '1713', '207', '1081',
                        '1577', '218', '1648', '2531', '2546', '358', '2556', '746', '2', '988', '779', '2545', '1485',
                        '1209', '1478', '1176', '1468', '976', '638', '979', '1743', '2537', '170', '1647', '2552', '9',
                        '1718', '1012', '1334', '1738', '876', '662', '2535', '1683', '1687', '1060', '607', '1179',
                        '865', '1378', '2532', '2551', '108', '530', '1512', '770', '237', '1719', '849', '1759',
                        '1636', '721', '1477', '1831', '1589', '0', '2553', '2550', '2533', '1503', '386', '1125',
                        '1579', '1695', '453', '1543', '685', '1833', '1069', '351', '1741', '1611', '1591', '1497',
                        '1404', '819']

    print('重要特恒长度为：', len(important_column))
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

        pred_PM25 = xgb_model.predict(pm_data[important_column])

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
        the_sum_score = 0
        the_sum_num = 0
        for i in range(800, 801):
            start_num = 840 * 750 + she_end_hour
            end_num = (1202) * 840
            # end_num=start_num+

            print('start_num:', start_num)
            print('end_num:', end_num)

            # pm_data = pm25_data.ix[int(start_num):int(end_num), :

            pm_data = pm25_data.ix[[i for i in range(start_num, end_num, 24)], :]

            pred_PM25 = xgb_model.predict(pm_data[important_column])

            # 分别计算   rmse_score  r2_score    mae_score    bias_score
            score = get_score(pred_PM25, pm_data[str(weidu_size - 1)])

            print(str(i) + '次，计算留出集合上损失得分：', score)
            # the_sum_score = the_sum_score + score
            # the_rmse_score = the_rmse_score + rmse_score
            # the_r2_score = the_r2_score + r2_score
            # the_mae_score = the_mae_score + mae_score
            # the_bias_score = the_bias_score + bias_score
            score_recore_list.append(score)

    print('1-3 小时上平均smape指标值：',np.mean(score_recore_list[0:3]))
    print('4-9 小时上平均smape指标值：',np.mean(score_recore_list[3:9]))
    print('9-16 小时上平均smape指标值：',np.mean(score_recore_list[9:16]))
    print('17-24 小时上平均smape指标值：',np.mean(score_recore_list[16:24]))
