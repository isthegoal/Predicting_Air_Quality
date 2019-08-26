#-*-coding:utf-8-*-
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import h5py
import  pickle
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler  # 这是标准化处理的语句，很方便，里面有标准化和反标准化。。
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor

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

def min_max_normalize(data):
    # 归一化       数据的归一化计算，这样计算之后结果能更加适合非树模型，  但是进行归一化之后怎么反归一化得看下
    # 数据量大，标准化慢
    df = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    # 做简单的平滑,试试效果如何
    return df



important_column=['135', '745', '227', '422', '607', '1297', '873', '1675', '1836', '302', '2377', '1489', '1203', '1488', '1663', '662', '1738', '2529', '1648', '1215', '1332', '719', '348', '1683', '831', '1094', '584', '2555', '961', '1333', '277', '95', '606', '1551', '1711', '1536', '4', '645', '652', '2560', '1735', '448', '657', '2531', '554', '1440', '2553', '722', '699', '170', '687', '1533', '1310', '1731', '468', '2151', '843', '1028', '1149', '1770', '1719', '324', '158', '1426', '1772', '767', '120', '784', '1021', '369', '444', '0', '1430', '1647', '681', '60', '48', '2538', '483', '1725', '314', '1548', '743', '1739', '697', '2528', '1749', '617', '518', '1750', '1707', '1459', '1746', '1763', '710', '793', '698', '2532', '2547', '1743', '1402', '2541', '254', '182', '1238', '1729', '122', '853', '2535', '290', '812', '410', '456', '2550', '1758', '2540', '432', '278', '1101', '264', '206', '566', '1537', '694', '375', '2551', '381', '1730', '691', '338', '783', '1394', '633', '602', '1736', '605', '592', '1784', '528', '2558', '674', '376', '993', '1748', '885', '1365', '650', '360', '1808', '1377', '2548', '2554', '2559', '33', '105', '1119', '1734', '1113', '532', '390', '433', '1578', '2543', '671', '1250', '2', '1575', '9', '2556', '1490', '374', '160', '1653', '756', '1756', '768', '1381', '134', '151', '350', '1718', '981', '1737', '470', '2544', '1671', '1716', '339', '37', '1742', '685', '849', '724', '265', '1077', '39', '1744', '996', '372', '686', '1366', '530', '1713', '688', '520', '604', '1755', '1768', '988', '1359', '1485', '805', '664', '1372', '1720', '1225', '1323', '1751', '1623', '1406', '969', '1741', '319', '1173', '1672', '1154', '420', '1346', '1572', '2536', '817', '1703', '2533', '1728', '2546', '700', '1200', '1767', '482', '1463', '535', '1089', '1096', '291', '723', '758', '408', '1251', '1745', '1012', '507', '2561', '1202', '384', '279', '2095', '262', '1629', '1775', '112', '1416', '820', '583', '1354', '2262', '51', '387', '1753', '537', '1466', '1423', '385', '1322', '1660', '1353', '1635', '1', '1689', '2545', '542', '1393', '1659', '1593', '2326', '1632', '2530', '1680', '1390', '231', '1565', '489', '1526', '717', '590', '1286', '1053', '1726', '1721', '84', '24', '727', '1378', '146', '693', '1701', '1754', '781', '1315', '733', '1065', '1779', '517', '97', '1335', '2539', '446', '1599', '945', '502', '218', '1418', '1237', '1515', '531', '1137', '1550', '362', '38', '242', '2184', '962', '861', '2197', '513', '1705', '1776', '458', '1041', '2542', '1727', '1563', '237', '1502', '1104', '948', '601', '775', '1221', '15', '1677', '230', '386', '2386', '2537', '2413', '1262', '1587', '266', '611', '87', '795', '1330', '916', '2534', '1083', '434', '734', '631', '585', '1477', '1668', '1699', '669', '195', '1740', '2549', '579', '626', '637', '1611', '1717', '1589', '1695', '1298', '2552', '928', '1806', '1723', '3', '643', '36', '506', '1684', '984', '1058', '1760', '649', '72', '2557', '398']

def model_nn_Model(train_air='pm25'):
    '''
        训练nn模型，并提取，倒数第二层的特征，特征的提取方法可参照：https://blog.csdn.net/hahajinbu/article/details/77982721
        进行最后一层的抽取方法是， 先训练一个nn模型model，但是要提前给每层都赋好层命名，   之后再简历一个Model,输入是上一个模型
        训练所使用到的数据，输出是上一个model的指定层名，最为输出，然后使用Model去做预测，得到输出那一层结果
        '''
    print('!!')
    f = h5py.File(
        './data/slice_eve_data/the_last_all__7_13_pca_com_800.h5',
        'r')
    # f['data'].value存放的是时间戳 上空间的流量数据
    data = f['data'].value
    weidu_size = data.shape[1]
    print('!')
    pm25_data_all = pd.DataFrame(data, columns=[str(i) for i in range(0, weidu_size)])
    print('00')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(pm25_data_all.ix[1:600000, :])
    pm25_data_all = pd.DataFrame(scaler.fit_transform(pm25_data_all.ix[1:600000, :]),
                                 columns=[str(i) for i in range(0, weidu_size)])

    # pm25_data_all = pm25_data_all.ix[1:300000, :]
    train_data = pm25_data_all[[str(i) for i in range(0, weidu_size - 1)]]
    train_flag = pm25_data_all[str((weidu_size - 1))]
    print('!!!')
    train_x, valid_x, train_y, valid_y = train_test_split(train_data, train_flag,
                                                          test_size=0.2, random_state=11)
    print('!!!!!')
    # print(train_x)

    ######################  1.书写之前的网络结构  ####################
    model = Sequential()
    model.add(Dense(activation='relu', units=2000, input_dim=400))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(activation='relu', units=800, name='Dense_2'))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(activation='tanh', units=1))

    optimizer = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=optimizer)

    # mc = ModelCheckpoint(filepath="./model/weights-improvement-{epoch:02d}-{val_auc:.2f}.h5", monitor='val_auc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=0)
    es = EarlyStopping(monitor='val_rmse', patience=5, verbose=1, mode='min')
    model.fit(x=train_x[important_column].values, y=train_y.values, batch_size=32, epochs=70,
              validation_data=(valid_x[important_column].values, valid_y.values), verbose=1, callbacks=[es])

    model_file = './data/save_model/nn_717_bn_' + train_air + '.model'
    model.save_weights(model_file, overwrite=True)

def model_nn_predict(train_air='pm25'):
    ######################  1.书写之前的网络结构  ####################
    model = Sequential()
    model.add(Dense(activation='relu', units=2000, input_dim=400))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(activation='relu', units=800, name='Dense_2'))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(activation='tanh', units=1))

    ###########################################################
    f = h5py.File(
        './data/slice_eve_data/the_last_all__7_13_pca_com_800.h5',
        'r')
    # f['data'].value存放的是时间戳 上空间的流量数据
    data = f['data'].value
    weidu_size = data.shape[1]

    model_file = 'data/save_model/nn_717_bn_' + train_air + '.model'
    model.load_weights(model_file)
    print('!', data)
    pm25_data_a = pd.DataFrame(data, columns=[str(i) for i in range(0, weidu_size)])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(pm25_data_a.ix[1:600000, :])

    pm25_data_all = pd.DataFrame(scaler.fit_transform(pm25_data_a.ix[630000:1010000, :]),
                                 columns=[str(i) for i in range(0, weidu_size)])

    data = pd.DataFrame(model.predict(pm25_data_all[important_column]))
    # score = model.evaluate(
    #    pm25_data_all[[str(i) for i in range(0, 3385)]], pm25_data_all['3385'], batch_size=80, verbose=1)

    the_sum_score = 0
    the_sum_num = 0
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(pm25_data_a.ix[1:600000, str(weidu_size - 1)].values.reshape(-1, 1))
    for i in range(0, 450):
        start_num = 840 * i
        end_num = (i + 1) * 840

        print('start_num:', start_num)
        print('end_num:', end_num)

        pm25_data = data.ix[int(start_num):int(end_num), :]


        print('转化前', pd.DataFrame(pm25_data).head())
        pm25_data = scaler.inverse_transform(pm25_data.values.reshape(-1, 1))
        print('转化后', pd.DataFrame(pm25_data).head())

        # print('第一个：',pm25_data)
        # print('第二个：',pm25_data_a.ix[int(500000+start_num):int(500000+end_num), '3385'])
        score = get_score(pm25_data,
                          pm25_data_a.ix[int(630000 + start_num):int(630000 + end_num), str(weidu_size - 1)].values)

        print(str(i) + '次，计算留出集合上损失得分：', score)
        the_sum_score = the_sum_score + score
        the_sum_num = the_sum_num + 1

    print('GBDT 平均得分：', the_sum_score / the_sum_num)

