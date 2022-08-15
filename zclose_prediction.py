#!/usr/bin/env python3.8

# -*- coding: utf-8 -*-
# @Time : 2022/8/12 14:34
# @Author : Steven Hu
# @FileName: zclose_predict.py
# @Software: PyCharm
# @E-mail : 107147256@qq.com

# # 项目需求：通过XGboost来实现预测zclose值

# # 导入各python包

# In[1]:


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.style.use('fivethirtyeight')

# # 导入数据

# In[3]:


all_data_set_path = r'csv_export\csv_data_1m_begin_40d_博22转债_SHSE.113650_SHSE.603916.csv'
all_data_set = pd.read_csv(all_data_set_path)

# In[4]:


print(all_data_set.head())

# In[9]:


print(all_data_set.info())  # 查看有多少数据及特征

# In[10]:


print(all_data_set.isnull().sum())  # 检查是否有空数据

# # 研究数据

# In[5]:


# 特征热力图 相关性分析
list_columns = all_data_set.columns
plt.figure(figsize=(15, 10))
sns.heatmap(all_data_set[list_columns].corr(), annot=True, fmt=".2f")
plt.show()

# In[6]:


# 对特征重要性进行排序
corr_1 = all_data_set.corr()
corr_1["zclose"].sort_values(ascending=False)

# # 数据预处理

# In[16]:


len_ = len(['zopen', 'zhigh', 'zlow', 'zclose']) * 3
col_numbers_drop = []
for i in range(3):
    col_numbers_drop.append(len_ + i)
print(col_numbers_drop)

# In[14]:


print(all_data_set.info())


# In[49]:


# 依据特征重要性，选择zlow zhigh zopen来进行预测zclose
# 数据选择t-n, ...., t-2 t-1 与 t 来预测未来 t+1
# 转换原始数据为新的特征列来进行预测,time_window可以用来调试用前几次的数据来预测
def series_to_supervised(data, time_window=3):
    data_columns = ['zopen', 'zhigh', 'zlow', 'zclose']
    data = data[data_columns]  # Note this is important to the important feature choice
    cols, names = list(), list()
    for i in range(time_window, -1, -1):
        # get the data
        cols.append(data.shift(i))  # 数据偏移量

        # get the column name
        if ((i - 1) <= 0):
            suffix = '(t+%d)' % abs(i - 1)
        else:
            suffix = '(t-%d)' % (i - 1)
        names += [(colname + suffix) for colname in data_columns]

    # concat the cols into one dataframe
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.index = data.index.copy()
    # remove the nan value which is caused by pandas.shift
    agg = agg.dropna(inplace=False)

    # remove unused col (only keep the "close" fied for the t+1 period)
    # Note col "close" place in the columns

    len_ = len(data_columns) * time_window
    col_numbers_drop = []
    for i in range(len(data_columns) - 1):
        col_numbers_drop.append(len_ + i)

    agg.drop(agg.columns[col_numbers_drop], axis=1, inplace=True)

    return agg


# In[56]:


all_data_set2 = all_data_set.copy()
all_data_set2["index"] = pd.to_datetime(all_data_set2["index"])  # 日期object: to datetime
all_data_set2.set_index("index", inplace=True, drop=True)  # 把index设为索引

# In[57]:


all_data_set2 = all_data_set2[116:]  # 这里把7月28日的数据全部删掉了，主要是数据缺失较多

# In[59]:


data_set_process = series_to_supervised(all_data_set2, 10)  # 取近10分钟的数据
print(data_set_process.columns.values)

# In[60]:


print(data_set_process.info())

# In[61]:


print(data_set_process.head())

# # 搭建模型XGboost

# In[ ]:
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_set_process)


# In[ ]:
train_size = int(len(data_set_process)*0.8)
test_size = len(data_set_process) - train_size
train_XGB, test_XGB = scaled_data[0:train_size,:],scaled_data[train_size:len(data_set_process),:]

train_XGB_X, train_XGB_Y = train_XGB[:,:(len(data_set_process.columns)-1)],train_XGB[:,(len(data_set_process.columns)-1)]
test_XGB_X, test_XGB_Y = test_XGB[:,:(len(data_set_process.columns)-1)],test_XGB[:,(len(data_set_process.columns)-1)]

# 算法参数
params = {
    'booster':'gbtree',
    'objective':'binary:logistic',  # 此处为回归预测，这里如果改成multi:softmax 则可以进行多分类
    'gamma':0.1,
    'max_depth':5,
    'lambda':3,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':3,
    'slient':1,
    'eta':0.1,
    'seed':1000,
    'nthread':4,
}

#生成数据集格式
xgb_train = xgb.DMatrix(train_XGB_X,label = train_XGB_Y)
xgb_test = xgb.DMatrix(test_XGB_X,label = test_XGB_Y)
num_rounds = 300
watchlist = [(xgb_test,'eval'),(xgb_train,'train')]

#xgboost模型训练
model_xgb = xgb.train(params,xgb_train,num_rounds,watchlist)

#对测试集进行预测
y_pred_xgb = model_xgb.predict(xgb_test)

mape_xgb = np.mean(np.abs(y_pred_xgb-test_XGB_Y)/test_XGB_Y)*100
print('XGBoost平均误差率为：{}%'.format(mape_xgb))  #平均误差率为1.1974%


# In[ ]:


# # 搭建模型LSTM网络

# In[62]:


# 注意这里要安装Tensorflow 和 Keras才能使用
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# In[ ]:

# split the train and test data
train_size = int(len(data_set_process) * 0.8)
test_size = len(data_set_process) - train_size
train_LSTM, test_LSTM = scaled_data[0:train_size, :], scaled_data[train_size:len(data_set_process), :]

train_LSTM_X, train_LSTM_Y = train_LSTM[:, :(len(data_set_process.columns) - 1)], train_LSTM[:,
                                                                                  (len(data_set_process.columns) - 1)]
test_LSTM_X, test_LSTM_Y = test_LSTM[:, :(len(data_set_process.columns) - 1)], test_LSTM[:,
                                                                               (len(data_set_process.columns) - 1)]

# reshape input to be [samples, time steps, features]
train_LSTM_X2 = np.reshape(train_LSTM_X, (train_LSTM_X.shape[0], 1, train_LSTM_X.shape[1]))
test_LSTM_X2 = np.reshape(test_LSTM_X, (test_LSTM_X.shape[0], 1, test_LSTM_X.shape[1]))

print(train_LSTM_X.shape, train_LSTM_Y.shape, test_LSTM_X.shape, test_LSTM_Y.shape)
print(train_LSTM_X2.shape, test_LSTM_X2.shape)

# creat and fit the LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(train_LSTM_X2.shape[1], train_LSTM_X2.shape[2])))
# model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss="mae", optimizer="Adam")
print(model.summary())

# In[ ]:

print("start to fit the model")
history = model.fit(train_LSTM_X2, train_LSTM_Y, epochs=50, batch_size=50, validation_data=(test_LSTM_X2, test_LSTM_Y),
                    verbose=2, shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

model.save('LSTM_model.h5')  # 这里保存模型，以便以后可以使用

# model的使用
# from tensorflow.keras.models import load_model
# del model  # 删除已存在的model
# model = load_model('LSTM_model.h5')

# make prediction
yPredict = model.predict(test_LSTM_X2)
print(yPredict.shape)

testPredict = scaler.inverse_transform(np.concatenate((test_LSTM_X, yPredict), axis=1))[:, -1:]
test_LSTM_Y2 = scaler.inverse_transform(np.concatenate((test_LSTM_X, test_LSTM_Y.reshape(len(test_LSTM_Y),1)), axis=1))[:, -1:]
print(testPredict.shape)
print(testPredict)

print("start calculate the mape")

mape = np.mean(np.abs(test_LSTM_Y2.flatten()-testPredict.flatten())/test_LSTM_Y2.flatten())*100
print('Test LSTM Score:%.6f MAPE' %(mape))

