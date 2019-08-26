import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from model.lightGBM import pre_train_lgb_model,model_lgb_train,model_lgb_predict
from model.XGBOOST import pre_model_xgb_train,model_xgb_train,model_xgb_predict
from model.GBDT import model_gbdt_train,model_gbdt_predict
from model.Adaboost import model_adaboost_train,model_adaboost_predict
from model.DNN import model_nn_Model,model_nn_predict
from model.stacking import model_stacking_train,model_stacking_predict
from model.linear_ensemble import model_linear_train,model_linear_predict



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="select model",type=str)
parser.add_argument("--mode", help="train or test",type=str)
args = parser.parse_args()
print('args.mode:',args.model)
print('args.mode:',args.mode)
select_model=args.model
select_mode=args.mode

########################   LightGBM model   ###################
###  pre-train LightGBM model to get important feature  name
#pre_train_lgb_model()
### train lightGBM model
if (select_model=='lightGBM' and select_mode=='train'):
    model_lgb_train()

### valid lightGBM model
if (select_model=='lightGBM' and select_mode=='test'):
    model_lgb_predict()

########################   Xgboost model   ###################

###  pre-train Xgboost model to get important feature  name
#pre_model_xgb_train()

### train Xgboost model
if (select_model=='Xgboost' and select_mode=='train'):
    model_xgb_train()

### valid Xgboost model
if (select_model=='Xgboost' and select_mode=='test'):
    model_xgb_predict()

########################   GBDT model   ###################

### train gbdt model
if (select_model=='Gbdt' and select_mode=='train'):
    model_gbdt_train()

### valid gbdt model
if (select_model=='Gbdt' and select_mode=='test'):
    model_gbdt_predict()

########################   Adaboost model   ###################

### train adaboost model
if (select_model=='Adaboost' and select_mode=='train'):
    model_adaboost_train()

### valid adaboost model
if (select_model=='Adaboost' and select_mode=='test'):
    model_adaboost_predict()

########################   DNN model   ###################

### train dnn model
if (select_model=='DNN' and select_mode=='train'):
    model_nn_Model()

### valid dnn model
if (select_model=='DNN' and select_mode=='test'):
    model_nn_predict()

########################   stacking model   ###################

### train stacking model
if (select_model=='stacking' and select_mode=='train'):
    model_stacking_train()

### valid stacking model
if (select_model=='stacking' and select_mode=='test'):
    model_stacking_predict()

########################   linear_ensemble model   ###################

### train linear_ensemble model
if (select_model=='linear_ensemble' and select_mode=='train'):
    model_linear_train()

### valid linear_ensemble model
if (select_model=='linear_ensemble' and select_mode=='test'):
    model_linear_predict()
