Introduction
------------
Codes for paper'A Feature Selection and Multi-model Fusion-based Approach of Predicting Air Quality'

Requirements
------------
- Python 3.5
- pandas 0.24.2
- numpy 
- h5py 
- sklearn 0.19.0
- matplotlib 
- xgboost 0.72.1
- lightgbm 2.1.1
- mlxtend 
- Keras 2.1.5

Contents
------------
- data: save data
- model: build and train model 
- process: Data preprocessing
- record: save feature important picture
- scripts: scripts to run processing or training

Instructions
------------
Pretrained model is in /data/save_model.

Run
```
python scripts/data_process_pipeline.py
```
to do data process.

Run
```
python scripts/model_pipeline.py --model lightGBM --mode train
python scripts/model_pipeline.py --model lightGBM --mode test
```
to train and test lightGBM model.

Run
```
python scripts/model_pipeline.py --model Xgboost --mode train
python scripts/model_pipeline.py --model Xgboost --mode test
```
to train and test Xgboost model.

Run
```
python scripts/model_pipeline.py --model linear_ensemble --mode train
python scripts/model_pipeline.py --model linear_ensemble --mode test
```
to train and test linear_ensemble model.


Run
```
python scripts/draw_feature_important.py
```
to draw feature important picture.


