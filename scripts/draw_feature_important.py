#-*-coding:utf-8-*-
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pandas as pd
import lightgbm as lgb
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import h5py
import  pickle
from pandas import DataFrame as DF
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.externals import joblib

print('—————————————       get feature name       ————————————————')
fea_name_list = []
for i in range(1, 145):
    for j in ['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph', 'weather', 'NO2', 'CO', 'SO2',
              'PM2.5', 'PM10', 'O3']:
        fea_name_list.append('before ' + str(145 - i) + ' hours ' + j)
for i in range(1, 801):
    fea_name_list.append('pca_' + str(i))
fea_name_list.extend(
    ['mean_pm25_1', 'mean_pm25_3', 'mean_pm25_5', 'max_pm25_1', 'max_pm25_3', 'max_pm25_5', 'min_pm25_1', 'min_pm25_3',
     'min_pm25_5', 'mean_pm10_1', 'mean_pm10_3',
     'mean_pm10_5', 'max_pm10_1', 'max_pm10_3', 'max_pm10_5', 'min_pm10_1', 'min_pm10_3', 'min_pm10_5', 'mean_O3_1',
     'mean_O3_3', 'mean_O3_5', 'max_O3_1', 'max_O3_3',
     'max_O3_5', 'min_O3_1', 'min_O3_3', 'min_O3_5', 'pm25_13', 'pm25_35', 'pm10_13', 'pm10_35', 'o3_13', 'o3_35'])
fea_name_list.append('the hour need predict')

def lgb_feature_draw():
    print('—————————————       LightGBM model       ————————————————')
    model_path = './data/save_model/lgb_best_yishen_0.4184.model'
    lgb_model = pickle.load(open(model_path, 'rb'))
    lgb.plot_importance(lgb_model,max_num_features=10)
    important_column=['135', '745', '227', '422', '607', '1297', '873', '1675', '1836', '302', '2377', '1489', '1203', '1488', '1663', '662', '1738', '2529', '1648', '1215', '1332', '719', '348', '1683', '831', '1094', '584', '2555', '961', '1333', '277', '95', '606', '1551', '1711', '1536', '4', '645', '652', '2560', '1735', '448', '657', '2531', '554', '1440', '2553', '722', '699', '170', '687', '1533', '1310', '1731', '468', '2151', '843', '1028', '1149', '1770', '1719', '324', '158', '1426', '1772', '767', '120', '784', '1021', '369', '444', '0', '1430', '1647', '681', '60', '48', '2538', '483', '1725', '314', '1548', '743', '1739', '697', '2528', '1749', '617', '518', '1750', '1707', '1459', '1746', '1763', '710', '793', '698', '2532', '2547', '1743', '1402', '2541', '254', '182', '1238', '1729', '122', '853', '2535', '290', '812', '410', '456', '2550', '1758', '2540', '432', '278', '1101', '264', '206', '566', '1537', '694', '375', '2551', '381', '1730', '691', '338', '783', '1394', '633', '602', '1736', '605', '592', '1784', '528', '2558', '674', '376', '993', '1748', '885', '1365', '650', '360', '1808', '1377', '2548', '2554', '2559', '33', '105', '1119', '1734', '1113', '532', '390', '433', '1578', '2543', '671', '1250', '2', '1575', '9', '2556', '1490', '374', '160', '1653', '756', '1756', '768', '1381', '134', '151', '350', '1718', '981', '1737', '470', '2544', '1671', '1716', '339', '37', '1742', '685', '849', '724', '265', '1077', '39', '1744', '996', '372', '686', '1366', '530', '1713', '688', '520', '604', '1755', '1768', '988', '1359', '1485', '805', '664', '1372', '1720', '1225', '1323', '1751', '1623', '1406', '969', '1741', '319', '1173', '1672', '1154', '420', '1346', '1572', '2536', '817', '1703', '2533', '1728', '2546', '700', '1200', '1767', '482', '1463', '535', '1089', '1096', '291', '723', '758', '408', '1251', '1745', '1012', '507', '2561', '1202', '384', '279', '2095', '262', '1629', '1775', '112', '1416', '820', '583', '1354', '2262', '51', '387', '1753', '537', '1466', '1423', '385', '1322', '1660', '1353', '1635', '1', '1689', '2545', '542', '1393', '1659', '1593', '2326', '1632', '2530', '1680', '1390', '231', '1565', '489', '1526', '717', '590', '1286', '1053', '1726', '1721', '84', '24', '727', '1378', '146', '693', '1701', '1754', '781', '1315', '733', '1065', '1779', '517', '97', '1335', '2539', '446', '1599', '945', '502', '218', '1418', '1237', '1515', '531', '1137', '1550', '362', '38', '242', '2184', '962', '861', '2197', '513', '1705', '1776', '458', '1041', '2542', '1727', '1563', '237', '1502', '1104', '948', '601', '775', '1221', '15', '1677', '230', '386', '2386', '2537', '2413', '1262', '1587', '266', '611', '87', '795', '1330', '916', '2534', '1083', '434', '734', '631', '585', '1477', '1668', '1699', '669', '195', '1740', '2549', '579', '626', '637', '1611', '1717', '1589', '1695', '1298', '2552', '928', '1806', '1723', '3', '643', '36', '506', '1684', '984', '1058', '1760', '649', '72', '2557', '398']
    lgb_important_column=[]
    for i in important_column:
        lgb_important_column.append(fea_name_list[int(i)])

    impt=lgb_model.feature_importances_

    plt.figure(figsize=(78,16))
    plt.subplot(142)
    lightGBM_col_imp=pd.DataFrame({'important_column':lgb_important_column,'importance':lgb_model.feature_importances_}).sort_values(by='importance',ascending=False)
    sns.barplot(x=lightGBM_col_imp['importance'][0:20],y=lightGBM_col_imp['important_column'][0:20],orient='h')
    plt.title('the lightGBM model')

    plt.savefig('./record/lightGBM_feature_important.jpg')
    plt.show()


def xgb_feature_draw():
    print('—————————————       xgboost model       ————————————————')
    from xgboost import XGBRegressor as XGBR
    model_path = './data/save_model/xgb_best_yishen_0.425.model'
    xgb_model = pickle.load(open(model_path, 'rb'))
    important_column = ['1322', '2184', '429', '1430', '2555', '768', '1473', '40', '336', '1739', '1734', '544', '24', '1251', '1540', '1274', '194', '693', '1757', '705', '1173', '432', '793', '362', '195', '1394', '279', '1284', '288', '458', '1429', '1549', '396', '669', '2542', '1505', '1772', '1642', '753', '1076', '1281', '1106', '542', '1442', '2540', '1469', '1716', '681', '518', '1457', '1269', '813', '1083', '1279', '158', '434', '1775', '871', '657', '1659', '1737', '691', '455', '555', '1707', '1567', '264', '1684', '1552', '2559', '1414', '1651', '1203', '1464', '820', '729', '1660', '1545', '99', '1438', '796', '1509', '2539', '384', '1730', '127', '1776', '1701', '1729', '2528', '1626', '1493', '12', '650', '647', '482', '211', '1161', '1107', '1675', '1311', '903', '314', '1736', '1623', '1197', '964', '645', '783', '494', '1299', '1541', '537', '549', '2541', '782', '182', '916', '1725', '348', '1238', '2139', '381', '206', '28', '146', '4', '444', '369', '2560', '2549', '1250', '1740', '1726', '890', '169', '1711', '1752', '1722', '225', '1557', '554', '597', '1149', '614', '183', '1447', '1599', '697', '398', '825', '177', '1342', '698', '734', '1239', '16', '1418', '372', '861', '877', '446', '147', '204', '928', '1413', '2561', '506', '1488', '2530', '578', '674', '2041', '837', '2543', '1320', '949', '468', '1395', '572', '1754', '302', '1226', '291', '1742', '148', '1131', '276', '1717', '710', '1671', '70', '1735', '1354', '1779', '873', '1514', '945', '1440', '1310', '626', '1435', '60', '1721', '2558', '2534', '590', '1663', '1491', '499', '1720', '1728', '633', '360', '1767', '1763', '495', '1495', '1613', '110', '324', '1713', '207', '1081', '1577', '218', '1648', '2531', '2546', '358', '2556', '746', '2', '988', '779', '2545', '1485', '1209', '1478', '1176', '1468', '976', '638', '979', '1743', '2537', '170', '1647', '2552', '9', '1718', '1012', '1334', '1738', '876', '662', '2535', '1683', '1687', '1060', '607', '1179', '865', '1378', '2532', '2551', '108', '530', '1512', '770', '237', '1719', '849', '1759', '1636', '721', '1477', '1831', '1589', '0', '2553', '2550', '2533', '1503', '386', '1125', '1579', '1695', '453', '1543', '685', '1833', '1069', '351', '1741', '1611', '1591', '1497', '1404', '819']

    xgb_important_column=[]
    for i in important_column:
        xgb_important_column.append(fea_name_list[int(i)])
    from matplotlib import pyplot as plt
    import seaborn as sns
    impt=xgb_model.feature_importances_

    plt.figure(figsize=(78,16))
    plt.subplot(142)
    xgboost_col_imp=pd.DataFrame({'important_column':xgb_important_column,'importance':xgb_model.feature_importances_}).sort_values(by='importance',ascending=False)
    sns.barplot(x=xgboost_col_imp['importance'][0:20],y=xgboost_col_imp['important_column'][0:20],orient='h')
    plt.title('the xgboost model')

    plt.savefig('./record/xgboost_feature_important.jpg')
    plt.show()

def gbdt_feature_draw():
    print('—————————————       gbdt model       ————————————————')
    model_path = './data/save_model/gbdt_best_yishen_0.4213.model'
    gbdt_model = pickle.load(open(model_path, 'rb'))

    important_column=['135', '745', '227', '422', '607', '1297', '873', '1675', '1836', '302', '2377', '1489', '1203', '1488', '1663', '662', '1738', '2529', '1648', '1215', '1332', '719', '348', '1683', '831', '1094', '584', '2555', '961', '1333', '277', '95', '606', '1551', '1711', '1536', '4', '645', '652', '2560', '1735', '448', '657', '2531', '554', '1440', '2553', '722', '699', '170', '687', '1533', '1310', '1731', '468', '2151', '843', '1028', '1149', '1770', '1719', '324', '158', '1426', '1772', '767', '120', '784', '1021', '369', '444', '0', '1430', '1647', '681', '60', '48', '2538', '483', '1725', '314', '1548', '743', '1739', '697', '2528', '1749', '617', '518', '1750', '1707', '1459', '1746', '1763', '710', '793', '698', '2532', '2547', '1743', '1402', '2541', '254', '182', '1238', '1729', '122', '853', '2535', '290', '812', '410', '456', '2550', '1758', '2540', '432', '278', '1101', '264', '206', '566', '1537', '694', '375', '2551', '381', '1730', '691', '338', '783', '1394', '633', '602', '1736', '605', '592', '1784', '528', '2558', '674', '376', '993', '1748', '885', '1365', '650', '360', '1808', '1377', '2548', '2554', '2559', '33', '105', '1119', '1734', '1113', '532', '390', '433', '1578', '2543', '671', '1250', '2', '1575', '9', '2556', '1490', '374', '160', '1653', '756', '1756', '768', '1381', '134', '151', '350', '1718', '981', '1737', '470', '2544', '1671', '1716', '339', '37', '1742', '685', '849', '724', '265', '1077', '39', '1744', '996', '372', '686', '1366', '530', '1713', '688', '520', '604', '1755', '1768', '988', '1359', '1485', '805', '664', '1372', '1720', '1225', '1323', '1751', '1623', '1406', '969', '1741', '319', '1173', '1672', '1154', '420', '1346', '1572', '2536', '817', '1703', '2533', '1728', '2546', '700', '1200', '1767', '482', '1463', '535', '1089', '1096', '291', '723', '758', '408', '1251', '1745', '1012', '507', '2561', '1202', '384', '279', '2095', '262', '1629', '1775', '112', '1416', '820', '583', '1354', '2262', '51', '387', '1753', '537', '1466', '1423', '385', '1322', '1660', '1353', '1635', '1', '1689', '2545', '542', '1393', '1659', '1593', '2326', '1632', '2530', '1680', '1390', '231', '1565', '489', '1526', '717', '590', '1286', '1053', '1726', '1721', '84', '24', '727', '1378', '146', '693', '1701', '1754', '781', '1315', '733', '1065', '1779', '517', '97', '1335', '2539', '446', '1599', '945', '502', '218', '1418', '1237', '1515', '531', '1137', '1550', '362', '38', '242', '2184', '962', '861', '2197', '513', '1705', '1776', '458', '1041', '2542', '1727', '1563', '237', '1502', '1104', '948', '601', '775', '1221', '15', '1677', '230', '386', '2386', '2537', '2413', '1262', '1587', '266', '611', '87', '795', '1330', '916', '2534', '1083', '434', '734', '631', '585', '1477', '1668', '1699', '669', '195', '1740', '2549', '579', '626', '637', '1611', '1717', '1589', '1695', '1298', '2552', '928', '1806', '1723', '3', '643', '36', '506', '1684', '984', '1058', '1760', '649', '72', '2557', '398']

    gbdt_important_column=[]
    for i in important_column:
        gbdt_important_column.append(fea_name_list[int(i)])

    from matplotlib import pyplot as plt
    import seaborn as sns
    impt=gbdt_model.feature_importances_

    plt.figure(figsize=(78,16))
    plt.subplot(142)
    gbdt_col_imp=pd.DataFrame({'important_column':gbdt_important_column,'importance':gbdt_model.feature_importances_}).sort_values(by='importance',ascending=False)
    sns.barplot(x=gbdt_col_imp['importance'][0:20],y=gbdt_col_imp['important_column'][0:20],orient='h')
    plt.title('the gbdt model')

    plt.savefig('./record/gbdt_feature_important.jpg')
    plt.show()

def adaboost_feature_draw():
    print('—————————————       adaboost model       ————————————————')

    adaboost_model = joblib.load('./data/save_model/adaboost_best_yishen_buzhi3.pkl')

    important_column=['135', '745', '227', '422', '607', '1297', '873', '1675', '1836', '302', '2377', '1489', '1203', '1488', '1663', '662', '1738', '2529', '1648', '1215', '1332', '719', '348', '1683', '831', '1094', '584', '2555', '961', '1333', '277', '95', '606', '1551', '1711', '1536', '4', '645', '652', '2560', '1735', '448', '657', '2531', '554', '1440', '2553', '722', '699', '170', '687', '1533', '1310', '1731', '468', '2151', '843', '1028', '1149', '1770', '1719', '324', '158', '1426', '1772', '767', '120', '784', '1021', '369', '444', '0', '1430', '1647', '681', '60', '48', '2538', '483', '1725', '314', '1548', '743', '1739', '697', '2528', '1749', '617', '518', '1750', '1707', '1459', '1746', '1763', '710', '793', '698', '2532', '2547', '1743', '1402', '2541', '254', '182', '1238', '1729', '122', '853', '2535', '290', '812', '410', '456', '2550', '1758', '2540', '432', '278', '1101', '264', '206', '566', '1537', '694', '375', '2551', '381', '1730', '691', '338', '783', '1394', '633', '602', '1736', '605', '592', '1784', '528', '2558', '674', '376', '993', '1748', '885', '1365', '650', '360', '1808', '1377', '2548', '2554', '2559', '33', '105', '1119', '1734', '1113', '532', '390', '433', '1578', '2543', '671', '1250', '2', '1575', '9', '2556', '1490', '374', '160', '1653', '756', '1756', '768', '1381', '134', '151', '350', '1718', '981', '1737', '470', '2544', '1671', '1716', '339', '37', '1742', '685', '849', '724', '265', '1077', '39', '1744', '996', '372', '686', '1366', '530', '1713', '688', '520', '604', '1755', '1768', '988', '1359', '1485', '805', '664', '1372', '1720', '1225', '1323', '1751', '1623', '1406', '969', '1741', '319', '1173', '1672', '1154', '420', '1346', '1572', '2536', '817', '1703', '2533', '1728', '2546', '700', '1200', '1767', '482', '1463', '535', '1089', '1096', '291', '723', '758', '408', '1251', '1745', '1012', '507', '2561', '1202', '384', '279', '2095', '262', '1629', '1775', '112', '1416', '820', '583', '1354', '2262', '51', '387', '1753', '537', '1466', '1423', '385', '1322', '1660', '1353', '1635', '1', '1689', '2545', '542', '1393', '1659', '1593', '2326', '1632', '2530', '1680', '1390', '231', '1565', '489', '1526', '717', '590', '1286', '1053', '1726', '1721', '84', '24', '727', '1378', '146', '693', '1701', '1754', '781', '1315', '733', '1065', '1779', '517', '97', '1335', '2539', '446', '1599', '945', '502', '218', '1418', '1237', '1515', '531', '1137', '1550', '362', '38', '242', '2184', '962', '861', '2197', '513', '1705', '1776', '458', '1041', '2542', '1727', '1563', '237', '1502', '1104', '948', '601', '775', '1221', '15', '1677', '230', '386', '2386', '2537', '2413', '1262', '1587', '266', '611', '87', '795', '1330', '916', '2534', '1083', '434', '734', '631', '585', '1477', '1668', '1699', '669', '195', '1740', '2549', '579', '626', '637', '1611', '1717', '1589', '1695', '1298', '2552', '928', '1806', '1723', '3', '643', '36', '506', '1684', '984', '1058', '1760', '649', '72', '2557', '398']

    adaboost_important_column=[]
    for i in important_column:
        adaboost_important_column.append(fea_name_list[int(i)])

    from matplotlib import pyplot as plt
    import seaborn as sns
    impt=adaboost_model.feature_importances_

    plt.figure(figsize=(78,16))
    plt.subplot(142)
    adaboost_col_imp=pd.DataFrame({'important_column':adaboost_important_column,'importance':adaboost_model.feature_importances_}).sort_values(by='importance',ascending=False)
    sns.barplot(x=adaboost_col_imp['importance'][0:20],y=adaboost_col_imp['important_column'][0:20],orient='h')
    plt.title('the adaboost model')

    plt.savefig('./record/adaboost_feature_important.jpg')
    plt.show()


### draw adaboost feature important picture
lgb_feature_draw()

### draw xgboost feature important picture
#xgb_feature_draw()

### draw gbdt feature important picture
#gbdt_feature_draw()	
	
### draw adaboost feature important picture
#adaboost_feature_draw()