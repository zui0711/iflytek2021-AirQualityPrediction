import numpy as np
import pandas as pd


def linear_interp(x, x_low, x_high, y_low, y_high):
    return (y_high - y_low) / (x_high - x_low) * (x - x_low) + y_low


def AQIi(x, name):
    if name == 'SO2':
        if x < 50:
            return linear_interp(x, 0, 50, 0, 50)
        elif x < 150:
            return linear_interp(x, 50, 150, 50, 100)
        elif x < 475:
            return linear_interp(x, 150, 475, 100, 150)
        elif x < 800:
            return linear_interp(x, 475, 800, 150, 200)
        elif x < 1600:
            return linear_interp(x, 800, 1600, 200, 300)
        elif x < 2100:
            return linear_interp(x, 1600, 2100, 300, 400)
        elif x < 2620:
            return linear_interp(x, 2100, 2620, 400, 500)
        else:
            return 500
    elif name == 'NO2':
        if x < 40:
            return linear_interp(x, 0, 40, 0, 50)
        elif x < 80:
            return linear_interp(x, 40, 80, 50, 100)
        elif x < 180:
            return linear_interp(x, 80, 180, 100, 150)
        elif x < 280:
            return linear_interp(x, 180, 280, 150, 200)
        elif x < 565:
            return linear_interp(x, 280, 565, 200, 300)
        elif x < 750:
            return linear_interp(x, 565, 750, 300, 400)
        elif x < 940:
            return linear_interp(x, 750, 940, 400, 500)
        else:
            return 500
    elif name == 'CO':
        if x < 2:
            return linear_interp(x, 0, 2, 0, 50)
        elif x < 4:
            return linear_interp(x, 2, 4, 50, 100)
        elif x < 14:
            return linear_interp(x, 4, 14, 100, 150)
        elif x < 24:
            return linear_interp(x, 14, 24, 150, 200)
        elif x < 36:
            return linear_interp(x, 24, 36, 200, 300)
        elif x < 48:
            return linear_interp(x, 36, 48, 300, 400)
        elif x < 60:
            return linear_interp(x, 48, 60, 400, 500)
        else:
            return 500
    elif name == 'O3' or name == 'O3_8h':
        if x < 100:
            return linear_interp(x, 0, 100, 0, 50)
        elif x < 160:
            return linear_interp(x, 100, 160, 50, 100)
        elif x < 215:
            return linear_interp(x, 160, 215, 100, 150)
        elif x < 265:
            return linear_interp(x, 215, 265, 150, 200)
        elif x < 800:
            return linear_interp(x, 265, 800, 200, 300)
        # elif x < 800:
        #     return linear_interp(x, 565, 750, 300, 400)
        # elif x < 60:
        #     return linear_interp(x, 750, 940, 400, 500)
        else:
            return 500
    elif name == 'PM10':
        if x < 50:
            return linear_interp(x, 0, 50, 0, 50)
        elif x < 150:
            return linear_interp(x, 50, 150, 50, 100)
        elif x < 250:
            return linear_interp(x, 150, 250, 100, 150)
        elif x < 350:
            return linear_interp(x, 250, 350, 150, 200)
        elif x < 420:
            return linear_interp(x, 350, 420, 200, 300)
        elif x < 500:
            return linear_interp(x, 420, 500, 300, 400)
        elif x < 600:
            return linear_interp(x, 500, 600, 400, 500)
        else:
            return 500
    elif name in ['PM25', 'PM2.5', 'PM2_5']:
        if x < 35:
            return linear_interp(x, 0, 35, 0, 50)
        elif x < 75:
            return linear_interp(x, 35, 75, 50, 100)
        elif x < 115:
            return linear_interp(x, 75, 115, 100, 150)
        elif x < 150:
            return linear_interp(x, 115, 150, 150, 200)
        elif x < 250:
            return linear_interp(x, 150, 250, 200, 300)
        elif x < 350:
            return linear_interp(x, 250, 350, 300, 400)
        elif x < 500:
            return linear_interp(x, 350, 500, 400, 500)
        else:
            return 500


if __name__=='__main__':

    df_train = pd.read_csv('初赛_训练集/保定2016年.csv')
    df_test = pd.read_csv('初赛_测试集/石家庄20160701-20170701.csv')
    df = pd.concat([df_train, df_test]).reset_index(drop=True)

    NAMES = ['SO2', 'CO', 'NO2', 'O3_8h', 'PM2.5', 'PM10']

    AQI_feats = []
    for name in NAMES:
        df['AQI_'+name] = df[name].map(lambda x: AQIi(x, name))
        AQI_feats.append('AQI_'+name)

    df['manual_AQI'] = df[AQI_feats].max(axis=1)
    df['AQI_sub'] = df['AQI'] - df['manual_AQI']

    df[['日期', 'IPRC', 'AQI', 'manual_AQI', 'AQI_sub']+AQI_feats].to_csv('AQI.csv', index=False)
