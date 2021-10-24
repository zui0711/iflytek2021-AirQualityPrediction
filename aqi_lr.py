import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from sklearn.metrics import mean_squared_error as mse, accuracy_score
from matplotlib.pyplot import plot, show, figure, title, legend
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
import time
from cal_AQI import AQIi


def get_score(y, y_pred):
    y_pred = (y_pred - np.min(y)) / (np.max(y) - np.min(y))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    # y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))
    return np.sqrt(mse(y, y_pred)) * 100


df_train = pd.read_csv('复赛数据集/train.csv')
df_test = pd.read_csv('复赛数据集/test.csv')

df = pd.concat([df_train, df_test]).reset_index(drop=True)

AQI_feats = []
AQI_div_feats = []
AQI_feats_pos1 = []
AQI_feats_neg1 = []
NAMES = ['SO2', 'CO', 'NO2', 'O3', 'PM2_5', 'PM10']
for name in NAMES:
    df['AQI_'+name] = df[name].map(lambda x: AQIi(x, name))
    AQI_feats.append('AQI_'+name)

df['manual_AQI'] = df[AQI_feats].max(axis=1)

df.loc[df['AQI'].isna(), 'AQI'] = df.loc[df['AQI'].isna(), 'manual_AQI']
df['AQI_mean'] = df[AQI_feats].mean(axis=1)
df['AQI_std'] = df[AQI_feats].std(axis=1)
df['AQI_min'] = df[AQI_feats].min(axis=1)
df['AQI_median'] = df[AQI_feats].median(axis=1)

for name in NAMES:
    df['AQI_div_'+name] = df['AQI_'+name] / df['AQI_mean']
    AQI_div_feats.append('AQI_div_'+name)

df['AQI_min/max'] = df['AQI_min']/df['manual_AQI']
df['AQI_max/min'] = df['AQI_min']/df['manual_AQI']
df['AQI_mean/max'] = df['AQI_mean']/df['manual_AQI']
df['AQI_median/max'] = df['AQI_median']/df['manual_AQI']
df['AQI_std/max'] = df['AQI_median']/df['manual_AQI']

df['AQI_min/mean'] = df['AQI_min']/df['AQI_mean']
df['AQI_max/mean'] = df['manual_AQI']/df['AQI_mean']

df['mean'] = df[NAMES].mean(axis=1)
df['std'] = df[NAMES].std(axis=1)
df['max'] = df[NAMES].max(axis=1)
df['min'] = df[NAMES].min(axis=1)
df['median'] = df[NAMES].median(axis=1)

# df['min/max'] = df['min']/df['max']
df['mean/max'] = df['mean']/df['max']
df['median/max'] = df['median']/df['max']
df['std/max'] = df['std']/df['max']
df['min/max'] = df['min']/df['max']

df['AQI_std_1'] = df[['AQI_SO2', 'AQI_CO', 'AQI_NO2', 'AQI_O3']].std(axis=1)

df['AQI_max_1'] = df[['AQI_SO2', 'AQI_CO', 'AQI_NO2', 'AQI_O3']].max(axis=1)
df['AQI_max_2'] = df[['AQI_PM2_5', 'AQI_PM10']].max(axis=1)

df['AQI_mean_1'] = df[['AQI_SO2', 'AQI_CO', 'AQI_NO2', 'AQI_O3']].mean(axis=1)
df['AQI_mean_2'] = df[['AQI_PM2_5', 'AQI_PM10']].mean(axis=1)

df['max_ratio1'] = df['AQI_max_1'] / df['manual_AQI']
df['max_ratio2'] = df['AQI_max_2'] / df['manual_AQI']

df['AQI_mean/max1'] = df['AQI_mean_1'] / df['AQI_max_1']
df['AQI_mean/max2'] = df['AQI_mean_2'] / df['AQI_max_2']

df['PM25_ratio'] = df['AQI_PM2_5'] / df['AQI_max_2']
df['PM10_ratio'] = df['AQI_PM10'] / df['AQI_max_2']

def get_level(x):
    if x <=50:
        return 0
    elif x <= 100:
        return 1
    elif x <=150:
        return 2
    elif x <= 200:
        return 3
    elif x <= 300:
        return 4
    else:
        return 5


def get_level_score(y, y_pred):
    y = list(map(get_level, y))
    y_pred = list(map(get_level, y_pred))
    return accuracy_score(y, y_pred)


df['quarter'] = pd.to_datetime(df['date']).dt.quarter
df['month'] = pd.to_datetime(df['date']).dt.month

data_onehot = pd.get_dummies(df['quarter'], prefix='quarter')
df = pd.concat([df, data_onehot], axis=1)
# data_onehot = pd.get_dummies(df['month'], prefix='month')
# df = pd.concat([df, data_onehot], axis=1)

df_train = df.loc[:len(df_train)-1].reset_index(drop=True)
df_test = df.loc[len(df_train):].reset_index(drop=True)

feats = ['PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
         'AQI_SO2', 'AQI_CO', 'AQI_NO2', 'AQI_O3', 'AQI_PM2_5', 'AQI_PM10', 'manual_AQI',
         'AQI_max_1', 'AQI_max_2',  # 'AQI_std_1',
         'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4',
         'PM25_ratio', 'PM10_ratio',
         'mean/max',
         'median/max',
         ]

df_train[['date', 'month', 'AQI']+feats].to_csv('temp/train.csv', index=False)
df_test[['date', 'month']+feats].to_csv('temp/test.csv', index=False)

add_df = df_train[df_train['date']!='2015/10/1']
df_train = pd.concat([df_train, add_df], axis=0).reset_index(drop=True)

print(feats)
score = []
pred_y = 0
seeds = [2021]
oof = np.zeros(len(df_train))
fold_num = 5
for seed in seeds:
    skf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train[feats], df_train['month'])):
        model = LinearRegression()
        model.fit(df_train.loc[train_idx, feats], df_train.loc[train_idx, 'AQI'])
        oof[val_idx] += (model.predict(df_train.loc[val_idx, feats]))/len(seeds)
        pred_y += (model.predict(df_test[feats]))/fold_num/len(seeds)

        score.append(np.sqrt(mse(df_train.loc[val_idx, 'AQI'],
                                 model.predict(df_train.loc[val_idx, feats]))))

print(score, np.mean(score))

df_train['oof'] = oof

# print(get_score(df_train['AQI'], oof))
print('RMSE: Lr--%.5f'%(
    np.sqrt(mse(df_train['AQI'], oof)))
      )


df_test['AQI'] = pred_y

# ans2_aqi = pd.read_csv('ans/lr_aqi_202109111610.csv')['AQI']
# print('GAP----------------------', np.sqrt(mse(df_test['AQI'], ans2_aqi))*100)

df_test['IPRC'] = 0
df_test[['date', 'AQI', 'IPRC']].to_csv('temp/aqi_lr.csv', index=False, header=['date', 'AQI', 'IPRC'])
