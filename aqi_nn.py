import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import os
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import time
import warnings
warnings.filterwarnings("ignore")
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(2021)

df_train = pd.read_csv('temp/train.csv')
df_test = pd.read_csv('temp/test.csv')
feats = ['PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
         'AQI_SO2', 'AQI_CO', 'AQI_NO2', 'AQI_O3', 'AQI_PM2_5', 'AQI_PM10', 'manual_AQI', 'AQI_max_1', 'AQI_max_2',]
LABEL = 'AQI'

scaler = preprocessing.StandardScaler()
# df_train[feats] = scaler.fit_transform(df_train[feats])

df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
df[feats] = scaler.fit_transform(df[feats])
df_train = df.loc[:len(df_train)-1].reset_index(drop=True)
df_test = df.loc[len(df_train):].reset_index(drop=True)

df = df_train.copy()

from torch.utils.data import Dataset, DataLoader
setup_seed(2021)
class myDataset(Dataset):
    def __init__(self, df, idx):
        self.train = df.loc[idx, feats].reset_index(drop=True)
        self.label = df.loc[idx, LABEL].reset_index(drop=True)

    def __getitem__(self, index):
        # print(index)
        return np.array(self.train.loc[index], dtype='float32'), np.array(self.label.loc[index], dtype='float32')

    def __len__(self):
        return len(self.label)


class SeqNet(nn.Module):
    def __init__(self):
        super(SeqNet, self).__init__()
        self.fc1 = nn.Linear(15, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_save_dir = 'drive/MyDrive/环境/'
def train_model(model, criterion, optimizer, lr_scheduler=None):
    total_iters=len(trainloader)
    print('--------------total_iters:{}'.format(total_iters))
    since = time.time()
    best_rmse = 1e7
    best_epoch = 0
    #
    iters = len(trainloader)
    for epoch in range(1,max_epoch+1):
        model.train(True)
        begin_time=time.time()
        # print('learning rate:{}'.format(optimizer.param_groups[-1]['lr']))
        print('Fold{} Epoch {}/{}'.format(fold+1,epoch, max_epoch))
        count=0
        train_loss = []
        for i, (inputs, labels) in (enumerate(trainloader)):
            # print(inputs)
            count+=1
            inputs = inputs.to(device)
            labels = labels.view(-1).to(device)
            # print(labels)

            out_linear = model(inputs).to(device).view(-1)
            # print(out_linear)
            loss = criterion(out_linear, labels)
            # print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 更新cosine学习率
            if lr_scheduler!=None:
                lr_scheduler.step(epoch + count / iters)
            if print_interval>0 and (i % print_interval == 0 or out_linear.size()[0] < train_batch_size):
                spend_time = time.time() - begin_time
                print(
                    ' Fold:{} Epoch:{}({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        fold+1,epoch, count, total_iters,
                        loss.item(), optimizer.param_groups[-1]['lr'],
                        spend_time / count * total_iters // 60 - spend_time // 60))
            #
            train_loss.append(loss.item())
        #lr_scheduler.step()
        val_rmse = val_model(model, criterion)
        print('valRmse: {:.4f}  '.format(val_rmse))
        model_out_path = model_save_dir+"/"+'fold_'+str(fold+1)+'_'+str(epoch) + '.pth'
        best_model_out_path = model_save_dir+"/"+'fold_'+str(fold+1)+'_best'+LABEL+'.pth'
        #save the best model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch=epoch
            torch.save(model.state_dict(), best_model_out_path)
            print("save best epoch: {} best val_rmse: {:.5f}".format(best_epoch,val_rmse))

    print('Fold{} Best RMSE: {:.5f} Best epoch:{}'.format(fold+1, best_rmse, best_epoch))
    return best_rmse

@torch.no_grad()
def val_model(model, criterion):
    # dset_sizes=len(val_dataset)
    model.eval()
    pres_list=[]
    labels_list=[]
    for data in val_loader:
        inputs, labels = data
        outputs = model(inputs)
        pres_list+=outputs.numpy().tolist()
        labels_list+=labels.numpy().tolist()
    #
    preds = np.array(pres_list)
    labels = np.array(labels_list)
    val_rmse = np.sqrt(metrics.mean_squared_error(labels, preds))
    return val_rmse


setup_seed(2021)
kf = KFold(n_splits=5, shuffle=True, random_state=2021)
skf = StratifiedKFold(n_splits=5)
gkf = GroupKFold(n_splits=5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()
print_interval = -1
max_epoch = 50

kfold_best_rmse = []
# print(len(df))
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['month'].values)):
    trainloader = torch.utils.data.DataLoader(
        myDataset(df, train_idx),
        batch_size=8, shuffle=True, pin_memory=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(
        myDataset(df, val_idx),
        batch_size=64, shuffle=False, pin_memory=True, num_workers=8)
    # print(trainloader)
    model = SeqNet()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5,
                                                                     last_epoch=-1)

    best_rmse = train_model(model, criterion, optimizer, lr_scheduler=scheduler)
    kfold_best_rmse.append(best_rmse)

print(kfold_best_rmse)
print(np.mean(kfold_best_rmse))


import torch
import numpy as np
def load_model(weight_path):
    print(weight_path)
    model = SeqNet()

    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def predict(test):
    ret = 0
    for i, model in enumerate(model_list):
        print('----model ', i)
        pres_list = []
        inputs = torch.from_numpy(test)#.cuda()
        outputs = model(inputs)
        ret += outputs.numpy()/5

    return ret


device=torch.device('cpu')
model_list=[]
for i in range(5):
    model_list.append(load_model('drive/MyDrive/环境/fold_'+str(i+1)+'_best'+LABEL+'.pth'))
df_test[LABEL] = predict(np.array(df_test[feats], dtype='float32'))
df_test['IPRC'] = 0
df_test[['date', 'AQI', 'IPRC']].to_csv('temp/aqi_nn.csv', index=False, header=['date', 'AQI', 'IPRC'])