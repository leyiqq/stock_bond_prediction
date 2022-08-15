



# 项目需求：通过XGboost来实现预测zclose值
# 导入各python包

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')

all_data_set_path = r'csv_export\csv_data_1m_begin_40d_博22转债_SHSE.113650_SHSE.603916.csv'
all_data_set = pd.read_csv(all_data_set_path)

print(all_data_set.head())
print(all_data_set.info()) #查看有多少数据及特征

print(all_data_set.isnull().sum()) #检查是否有空数据