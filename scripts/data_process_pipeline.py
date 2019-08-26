import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from process.data_merge_process.merge_for_newdata.merge_grid_18_lon_lat import get_grid_lon_lat
from process.data_merge_process.merge_for_newdata.merge_meo_lon_lat import get_meo_lon_lat

from process.data_merge_process.data_process.merge_meo_grid import ProcessAqData
from process.data_merge_process.data_process.merge_aq_meo import merge_am
from process.data_merge_process.data_process.merge_aq_grid_meo import merge_agm

from process.data_process_together import data_process_toge
from process.trans_weather_to_num import trans_weather_to_num
from process.deal_weather_predition import deal_weather_pred
from process.do_simply_feature import do_ori_weather_pred_data

from process.fill_missing.fill_missing import missing_fill_air_pul,check_data_null

from process.do_slice_data import do_slice,do_flag_slice




########################   first step：Data merge   ###################

#  Initial merger
get_grid_lon_lat()
get_meo_lon_lat()

#  Merging again
ProcessAqData()
merge_am()
merge_agm()

########################   Second step：Data preprocessing  ################

# exceptional handling
data_process_toge()  

# Handling of the weather
trans_weather_to_num() 

# Weather forecast data preprocessing
deal_weather_pred() 

# Handling the lack of time 
missing_fill_air_pul()

#  Feature processing for weather forecast data
do_ori_weather_pred_data() 

#  Check for missing items and do interpolation
check_data_null()

########################   third step：Feature engineering  ################
#  Building a timed sliding window
do_slice()
#  Build timing characteristics
do_flag_slice()
