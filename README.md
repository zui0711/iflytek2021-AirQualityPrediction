# iflytek2021-AirQualityPrediction
2021年讯飞开发者大赛空气质量赛道Top2解决方案

感谢队友@许垒

算法源代码包含6个.py文件，以及3个文件夹
其中

cal_AQI.py为计算标准AQI的库
aqi_lr.py为使用LinearRegression预测AQI代码，在temp文件夹下生成aqi_lr.csv/train.csv/test.csv
aqi_nn.py为使用神经网络预测AQI代码，在temp文件夹下生成aqi_nn.csv
aqi_xgb.py为使用xgboost预测AQI代码，在temp文件夹下生成aqi_xgb.csv
iprc.py为预测IPRC代码，在temp文件夹下生成iprc.csv
merge.py为联合所有中间结果，并在ans文件夹下生成最终提交文件sub.csv

temp文件夹存储中间结果，ans文件夹存储最终提交文件，复赛数据集文件夹为复赛原始数据

复现时，依次运行
aqi_lr.py
aqi_nn.py
aqi_xgb.py
iprc.py
merge.py
即可得到最终提交结果
