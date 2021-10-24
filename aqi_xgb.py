import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from xgboost import plot_importance

def score(y_true,y_pred):
    #对预测值和实际值都执行归一化
    norm_y_true = (y_true - y_true.min())/(y_true.max() - y_true.min())
    norm_y_pred = (y_pred - y_true.min())/(y_true.max() - y_true.min())
    return (mean_squared_error(y_true=norm_y_true,y_pred=norm_y_pred) ** 0.5) * 100

def rmse(y_true,y_pred):
    return mean_squared_error(y_true=y_true,y_pred=y_pred) ** 0.5

"""
读取csv文件
"""
train_datas = pd.read_csv("复赛数据集/train.csv")
test_datas = pd.read_csv("复赛数据集/test.csv")
sb_data = pd.read_csv("复赛数据集/test.csv").date
submit_sp = pd.read_csv("复赛数据集/submit_sample.csv")

train_datas.columns = ["date","AQI","iprc","pm25","pm10","so2","no2","co","o3_8h"]
test_datas.columns = ["date","pm25","pm10","so2","no2","co","o3_8h"]

"""
处理时间格式数据---训练集：
空气质量和季节的相关系比较密切。
"""
train_datas['date'] = pd.to_datetime(train_datas.date,format='%Y-%m-%d')
train_datas['month'] = train_datas['date'].dt.month
train_datas['quarter'] = train_datas['date'].dt.quarter
train_datas_dates = train_datas[['date']]
#删除时间
train_datas.drop(["date"],axis=1,inplace=True)

"""
处理时间格式数据---测试集
"""
test_datas['date'] = pd.to_datetime(test_datas.date,format='%Y-%m-%d')
test_datas['month']=test_datas['date'].dt.month
test_datas['quarter'] = test_datas['date'].dt.quarter
#删除时间
test_datas.drop(["date"],axis=1,inplace=True)

df_datas = pd.concat([train_datas,test_datas],axis = 0)
"""
计算每个特征的IAQ值
"""
df_datas.loc[df_datas.pm25 > 500,'pm25'] = 500
def cul_pm25_iaqi(x):
    result = x
    if x >= 350:
        result = (100/150)*(x - 350)+400
    elif x >= 250:
        result = (x-250)+300
    elif x >= 150:
        result = (x-150)+200
    elif x >= 115:
        result = (50/35)*(x-115)+150
    elif x >= 75:
        result = (50/40)*(x-75)+100
    elif x >= 35:
        result = (50/40)*(x-35)+50
    elif x >= 0:
        result = (50/35)*(x-0)+0
    return result
df_datas["pm25_aqi"] = df_datas['pm25'].apply(cul_pm25_iaqi)

df_datas.loc[df_datas.pm10 > 600,'pm10'] = 600
def cul_pm10_iaqi(x):
    result = x
    if x >= 500:
        result = (x - 500)+400
    elif x >= 420:
        result = (100/80)*(x-420)+300
    elif x >= 350:
        result = (100/70)*(x-350)+200
    elif x >= 250:
        result = (50/100)*(x-250)+150
    elif x >= 150:
        result = (50/100)*(x-150)+100
    elif x >= 50:
        result = (50/100)*(x-50)+50
    elif x >= 0:
        result = (50/50)*(x-0)+0
    return result
df_datas["pm10_aqi"] = df_datas['pm10'].apply(cul_pm10_iaqi)

def cul_so2_iaqi(x):
    result = x
    if x >= 2100:
        result = (100/520)*(x-2100)+400
    elif x >= 1600:
        result = (100/500)*(x-1600)+300
    elif x >= 800:
        result = (100/800)*(x-800)+200
    elif x >= 475:
        result = (50/325)*(x-475)+150
    elif x >= 150:
        result = (50/325)*(x-150)+100
    elif x >= 50:
        result = (50/100)*(x-50)+50
    elif x >= 0:
        result = (50/50)*(x-0)+0
    return result
df_datas["so2_aqi"] = df_datas['so2'].apply(cul_so2_iaqi)

def cul_co_iaqi(x):
    result = x
    if x >= 48:
        result = (100/12)*(x-48)+400
    elif x >= 36:
        result = (100/12)*(x-36)+300
    elif x >= 24:
        result = (100/12)*(x-24)+200
    elif x >= 14:
        result = (50/10)*(x-14)+150
    elif x >= 4:
        result = (50/10)*(x-4)+100
    elif x >= 2:
        result = (50/2)*(x-2)+50
    elif x >= 0:
        result = (50/2)*(x-0)+0
    return result
df_datas["co_aqi"] = df_datas['co'].apply(cul_co_iaqi)

def cul_no2_iaqi(x):
    result = x
    if x >= 750:
        result = (100/190)*(x-750)+400
    elif x >= 565:
        result = (100/185)*(x-565)+300
    elif x >= 280:
        result = (100/285)*(x-280)+200
    elif x >= 180:
        result = (50/100)*(x-180)+150
    elif x >= 80:
        result = (50/100)*(x-80)+100
    elif x >= 40:
        result = (50/40)*(x-40)+50
    elif x >= 0:
        result = (50/40)*(x-0)+0
    return result
df_datas["no2_aqi"] = df_datas['no2'].apply(cul_no2_iaqi)

def cul_o3_8h_iaqi(x):
    result = x
    if x >= 265:
        result = (100/535)*(x-265)+200
    elif x >= 215:
        result = (50/50)*(x-215)+150
    elif x >= 160:
        result = (50/55)*(x-160)+100
    elif x >= 100:
        result = (50/60)*(x-100)+50
    elif x >= 0:
        result = (50/100)*(x-0)+0
    return result
df_datas["o3_8h_aqi"] = df_datas['o3_8h'].apply(cul_o3_8h_iaqi)

"""
对污染物的AQI构造一些有效特征-----这些特征对于预测AQI
"""
df_datas["AQI_max"] =  round(df_datas[["pm25_aqi","pm10_aqi"]].max(axis = 1))

df_datas["AQI_max_org"] =  df_datas[["pm25_aqi","pm10_aqi","so2_aqi","co_aqi","no2_aqi","o3_8h_aqi"]].max(axis = 1)
df_datas["AQI_mean_one"] =  df_datas[["pm25_aqi","pm10_aqi"]].mean(axis = 1)
df_datas["AQI_std_one"] =  df_datas[["pm25_aqi","pm10_aqi"]].std(axis = 1)

df_datas["AQI_mean_two"] =  df_datas[["so2_aqi","co_aqi","no2_aqi","o3_8h_aqi"]].mean(axis = 1)
df_datas["AQI_std_two"] =  df_datas[["so2_aqi","co_aqi","no2_aqi","o3_8h_aqi"]].std(axis = 1)

"""
对季节进行one_hot编码
"""
df_datas["quarter"] = df_datas.quarter.astype("str")
quarter_df = df_datas.quarter.str.get_dummies()
quarter_df.columns = ["quarter_1","quarter_2","quarter_3","quarter_4"]
df_datas = pd.concat([df_datas,quarter_df],axis = 1)
df_datas.drop(["quarter"],axis=1,inplace=True)

f1 = "no2"
f2 = "o3_8h"
colname = '{}_add_{}'.format(f1, f2)
df_datas[colname] = df_datas[f1].values + df_datas[f2].values

"""
消除冗余特征
"""
df_datas.drop(['pm25_aqi', 'pm10_aqi', 'so2_aqi',
       'co_aqi', 'no2_aqi', 'o3_8h_aqi'],axis = 1,inplace=True)

"""
分离训练集和测试集 
"""
train_datas = df_datas[df_datas.AQI.notnull()]
test_datas = df_datas[df_datas.AQI.isna()].drop(['AQI'],axis = 1)

xgb_model = xgb.XGBRegressor(max_depth=3,
                            learning_rate=0.09,
                            n_estimators=1000,
                            subsample=0.5)

def show_plt_AQI(oof_train):
    """
    可视化，在验证集上，预测的拟合程度
    """
    for i in range(0,490,98):
        plt.figure(figsize=(20,5),facecolor = "w")
        plt.plot(train_datas[i:i+99].AQI,label="label")
        plt.plot(pd.Series(oof_train)[i:i+99],label = "AQI")
        plt.grid(color="#666A6D")
        plt.legend()
        plt.show()

repeats_time_seeds = [42,520,1314,2021,25]
repeats_times = len(repeats_time_seeds)

def Train_AQI_XGBPredict(train_datas,test_datas):
    n_fold = 5
    oof_train = np.zeros((train_datas.shape[0],))
    oof_test = np.zeros((test_datas.shape[0],))
    train_score = 0
    train_rmse = 0
    for r, seed in enumerate(repeats_time_seeds):
        oof_train_r = np.zeros((train_datas.shape[0],))
        oof_test_r = np.zeros((test_datas.shape[0],))
        folds = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        for numbers, (train_index, valid_index) in enumerate(folds.split(train_datas, train_datas['month'])):
            print("\n")
            print("------------>>>>>第%d次交叉验证：" % (numbers + 1))
            x_train = train_datas.drop(['iprc', 'AQI', 'month'], axis=1).iloc[train_index]
            x_valid = train_datas.drop(['iprc', 'AQI', 'month'], axis=1).iloc[valid_index]

            y_train = train_datas.iloc[train_index]['AQI'] - train_datas.iloc[train_index]['AQI_max']
            y_valid = train_datas.iloc[valid_index]['AQI'] - train_datas.iloc[valid_index]['AQI_max']

            xgb_model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)], eval_metric='rmse',
                          early_stopping_rounds=100, verbose=100)

            valid_pre = xgb_model.predict(x_valid)
            oof_train_r[valid_index] = valid_pre + train_datas.iloc[valid_index]['AQI_max']
            oof_test_r = oof_test_r + (
                        xgb_model.predict(test_datas.drop(['iprc', 'month'], axis=1)) + test_datas['AQI_max']) / n_fold

        train_rmse = train_rmse + rmse(train_datas.AQI, oof_train_r) / repeats_times
        oof_train = oof_train + oof_train_r / repeats_times
        oof_test = oof_test + oof_test_r / repeats_times
    print("..............训练集RMSE：", train_rmse, "..............")
    """
    可视化，在训练集上，预测的拟合程度
    """
    # show_plt_AQI(oof_train)
    return oof_test

AQI_pre = Train_AQI_XGBPredict(train_datas,test_datas)

"""
填充测试集的AQI
"""
df_datas.loc[df_datas.AQI.isna(),"AQI"] = AQI_pre

lr_result = pd.DataFrame({"data":sb_data,
                          "AQI":df_datas[df_datas.iprc.isna()].AQI,
                          "IPRC":0})
"""
保存预测结果
"""
lr_result.to_csv("temp/aqi_xgb.csv",index=False)
