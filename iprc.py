import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def score(y_true,y_pred):
    #对预测值和实际值都执行归一化
    norm_y_true = (y_true - y_true.min())/(y_true.max() - y_true.min())
    norm_y_pred = (y_pred - y_true.min())/(y_true.max() - y_true.min())
    return (mean_squared_error(y_true=norm_y_true,y_pred=norm_y_pred) ** 0.5) * 100
    
def rmse(y_true,y_pred):
    return mean_squared_error(y_true=y_true,y_pred=y_pred) ** 0.5
    
# train_datas = pd.read_csv(r"E:\Downlods\环境空气质量评价挑战赛_复赛数据集2\train.csv")
# test_datas = pd.read_csv(r"E:\Downlods\环境空气质量评价挑战赛_复赛数据集2\test.csv")

train_datas = pd.read_csv(r"复赛数据集/train.csv")
test_datas = pd.read_csv(r"复赛数据集/test.csv")

train_datas.columns = ["date","AQI","iprc","pm25","pm10","so2","no2","co","o3_8h"]
test_datas.columns = ["date","pm25","pm10","so2","no2","co","o3_8h"]
sb_data = test_datas.date
submit_sp = pd.read_csv(r"复赛数据集/submit_sample.csv")

"""
处理时间格式数据---训练集：
空气质量和季节的相关系比较密切。
"""
train_datas['date'] = pd.to_datetime(train_datas.date,format='%Y-%m-%d')
train_datas['month'] = train_datas['date'].dt.month
train_datas_dates = train_datas[['date']]
#删除时间
train_datas.drop(["date"],axis=1,inplace=True)

"""
处理时间格式数据---测试集
"""
test_datas['date'] = pd.to_datetime(test_datas.date,format='%Y-%m-%d')
test_datas['month']=test_datas['date'].dt.month
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
预测iprc
"""
df_datas = df_datas[['iprc', 'pm25', 'pm10', 'so2', 'no2', 'co', 'o3_8h', 'month',
       'pm25_aqi', 'pm10_aqi', 'so2_aqi', 'co_aqi', 'no2_aqi', 'o3_8h_aqi']]
       
"""
对污染物的AQI构造一些统计特征
"""
df_datas["AQI_max"] =  df_datas[["pm25_aqi","pm10_aqi","so2_aqi","co_aqi","no2_aqi","o3_8h_aqi"]].max(axis = 1)
df_datas["AQI_mean"] =  df_datas[["pm25_aqi","pm10_aqi","so2_aqi","co_aqi","no2_aqi","o3_8h_aqi"]].mean(axis = 1)
df_datas["AQI_min"] =  df_datas[["pm25_aqi","pm10_aqi","so2_aqi","co_aqi","no2_aqi","o3_8h_aqi"]].min(axis = 1)
df_datas["AQI_median"] =  df_datas[["pm25_aqi","pm10_aqi","so2_aqi","co_aqi","no2_aqi","o3_8h_aqi"]].median(axis = 1)
df_datas["AQI_std"] =  df_datas[["pm25_aqi","pm10_aqi","so2_aqi","co_aqi","no2_aqi","o3_8h_aqi"]].std(axis = 1)

"""
对于预测iprc，需要重新构造污染等级特征
"""
def cul_leve(aqi_val):
    leve = ""
    if aqi_val > 300:
        leve = "严重污染"
    elif aqi_val > 200:
        leve = "重度污染"
    elif aqi_val > 150:
        leve = "中度污染"
    elif aqi_val > 100:
        leve = "轻度污染"
    elif aqi_val > 50:
        leve = "良"
    elif aqi_val >= 0:
        leve = "优"
    return leve
    
df_datas["leve"] = df_datas['AQI_max'].apply(cul_leve)

df_datas.loc[df_datas.AQI_max > 300,"leve"] = "重度污染"
"""
对污染等级one_hot编码----训练集
"""
leve_df = df_datas.leve.str.get_dummies()
leve_df.columns = ["A","B","C","D","E"]
df_datas = pd.concat([df_datas,leve_df],axis = 1)
df_datas.drop(["leve"],axis=1,inplace=True)

"""
----------强特1
"""
df_datas["AQI_mean_div_maxMEAN"] = df_datas["AQI_mean"] / df_datas.groupby(["A","B","C","D","E"])["AQI_max"].transform('mean')
"""
----------强特2
"""
df_datas["AQI_mean_div_minMEAN"] = df_datas["AQI_mean"] / (df_datas.groupby(["A","B","C","D","E"])["AQI_min"].transform('mean'))
"""
----------强特3
"""
df_datas["AQI_mean_div_meanMEAN"] = df_datas["AQI_mean"] / df_datas.groupby(["A","B","C","D","E"])["AQI_mean"].transform('mean')
"""
----------强特4
"""
df_datas["AQI_meaian_div_medianMEAN"] = df_datas["AQI_mean"] / df_datas.groupby(["A","B","C","D","E"])["AQI_median"].transform('mean')

df_datas.drop(['pm25_aqi', 'pm10_aqi', 'so2_aqi',
       'co_aqi', 'no2_aqi', 'o3_8h_aqi'],axis = 1,inplace=True)
       
"""
分离训练集和测试集 
"""
train_datas = df_datas[df_datas.iprc.notnull()]
test_datas = df_datas[df_datas.iprc.isna()].drop(['iprc',"month"],axis = 1)       

repeats_time_seeds = [42,520,1314,2021,25]
repeats_times = len(repeats_time_seeds)

repeats_time_seeds = [42,520,1314,2021,25]
repeats_times = len(repeats_time_seeds)

def show_plt(oof_train,lens = 160):
    """
    可视化，在验证集上，预测的拟合程度
    """
    plt.figure(figsize=(20,10),facecolor = "w")
    ax_one = plt.subplot(2,1,1)
    ax_one.plot(oof_train[:lens],label = "pre")
    ax_one.plot(list(train_datas[:lens].iprc),label="label")
    ax_one.grid(color="#666A6D")
    ax_one.legend()
    ax_two = plt.subplot(2,1,2)
    ax_two.plot(oof_train[-lens:],label = "pre")
    ax_two.plot(list(train_datas[-lens:].iprc),label="label")
    ax_two.legend()
    ax_two.grid(color="#666A6D")
    plt.show()

"""
使用全数据集
"""

def Train_iprePredict(train_datas, test_datas):
    """
    使用全数据集
    """
    oof_train = np.zeros((train_datas.shape[0],))
    oof_test = np.zeros((test_datas.shape[0],))
    train_score = 0
    train_rmse = 0
    for r, seed in enumerate(repeats_time_seeds):
        x_train = train_datas.drop(['iprc', "month"], axis=1).values
        y_train = train_datas['iprc'].values
        # 开始拟合
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        oof_train = oof_train + lr.predict(x_train) / repeats_times
        oof_test = oof_test + lr.predict(test_datas.values) / repeats_times

    train_score = score(train_datas.iprc, oof_train)
    train_rmse = rmse(train_datas.iprc, oof_train)
    print("\n训练集Score：", train_score)
    print("训练集Rmse：", train_rmse)
    # show_plt(oof_train)
    return oof_test

iprc_pre = Train_iprePredict(train_datas, test_datas)


#修正处理
test_datas["AQI"] = 0
test_datas['iprc'] = iprc_pre
test_datas.loc[test_datas.AQI_max > 300, "iprc"] = test_datas.loc[test_datas.AQI_max > 300, "iprc"] * 1.07
lr_result = pd.DataFrame({"data":sb_data,
                          "AQI":test_datas.AQI,
                          "IPRC":test_datas.iprc})

"""
保存预测结果
"""
lr_result.to_csv("ans/iprc.csv",index=False)
