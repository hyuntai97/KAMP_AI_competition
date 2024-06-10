import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.model_selection import train_test_split
import re

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', seq_len=50,
                 data_path='competition_data.csv', scale=True, timeenc=0):
        
        # info 
        self.seq_len = seq_len

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        unlabeled(0), abnormal(-1), normal(1)
        '''
        df_raw['TAG'] = df_raw['TAG'].apply(lambda x: 1 if x=='OK' else (-1 if x=='NG' else 0))
        label = df_raw[['TAG']].values

        '''
        df_raw.columns: ['date', ...(other features)]
        '''
        cols = ['STD_DT', 'MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP']
        cols.remove('STD_DT')
        df_raw = df_raw[['STD_DT'] + cols]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['STD_DT']][border1:border2]
        df_stamp['STD_DT'] = pd.to_datetime(df_stamp.STD_DT)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.STD_DT.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.STD_DT.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.STD_DT.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.STD_DT.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['STD_DT'], 1).values
        elif self.timeenc == 1: # 아무것도 안들어가는 걸로 수정 필요 !
            df_stamp['month'] = df_stamp.STD_DT.apply(lambda row: row.month, 1)
            data_stamp = df_stamp.drop(['STD_DT'], 1).values

        self.data_x = data[border1:border2]
        self.data_y = label[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        
        s_end = s_begin + self.seq_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        return seq_x, seq_y, seq_x_mark 

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
        

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='test', seq_len=50,
                 data_path='competition.csv', scale=True, timeenc=0):
        
        # info 
        self.seq_len = seq_len

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc
        self.stride = seq_len 

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        unlabeled(0), abnormal(-1), normal(1)
        '''
        df_raw['TAG'] = df_raw['TAG'].apply(lambda x: 1 if x=='OK' else (-1 if x=='NG' else 0))
        label = df_raw[['TAG']].values

        '''
        df_raw.columns: ['date', ...(other features)]
        '''
        cols = ['STD_DT', 'MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP']
        cols.remove('STD_DT')
        df_raw = df_raw[['STD_DT'] + cols]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['STD_DT']][border1:border2]
        df_stamp['STD_DT'] = pd.to_datetime(df_stamp.STD_DT)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.STD_DT.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.STD_DT.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.STD_DT.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.STD_DT.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['STD_DT'], 1).values
        elif self.timeenc == 1: # 아무것도 안들어가는 걸로 수정 필요 !
            df_stamp['month'] = df_stamp.STD_DT.apply(lambda row: row.month, 1)
            data_stamp = df_stamp.drop(['STD_DT'], 1).values

        self.data_x = data[border1:border2]
        self.data_y = label[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index * self.stride
        
        s_end = s_begin + self.seq_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        return seq_x, seq_y, seq_x_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len)//self.stride  + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    
class Dataset_Custom_TCN(Dataset):
    def __init__(self, root_path, flag='train', seq_len=50,
                 data_path='competition_data.csv', scale=True, timeenc=0):
        
        # info 
        self.seq_len = seq_len

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        unlabeled(0), abnormal(-1), normal(1)
        '''
        df_raw['TAG'] = df_raw['TAG'].apply(lambda x: 1 if x=='OK' else (-1 if x=='NG' else 0))
        label = df_raw[['TAG']].values

        '''
        df_raw.columns: ['date', ...(other features)]
        '''
        cols = ['STD_DT', 'MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP']
        cols.remove('STD_DT')
        df_raw = df_raw[['STD_DT'] + cols]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['STD_DT']][border1:border2]
        df_stamp['STD_DT'] = pd.to_datetime(df_stamp.STD_DT)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.STD_DT.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.STD_DT.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.STD_DT.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.STD_DT.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['STD_DT'], 1).values
        elif self.timeenc == 1: # 아무것도 안들어가는 걸로 수정 필요 !
            df_stamp['month'] = df_stamp.STD_DT.apply(lambda row: row.month, 1)
            data_stamp = df_stamp.drop(['STD_DT'], 1).values

        self.data_x = data[border1:border2]
        self.data_y = label[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        
        s_end = s_begin + self.seq_len
        
        seq_x = self.data_x[s_begin:s_end].reshape(-1, self.seq_len)
        seq_y = self.data_y[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end].reshape(-1, self.seq_len)
        
        return seq_x, seq_y, seq_x_mark 

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

# TCN 기반 dataloader
class Dataset_Custom_tr_split(Dataset):
    def __init__(self, root_path, flag='train', seq_len=50,
                 data_path='competition_data.csv', scale=True, timeenc=0):
        
        # info 
        self.seq_len = seq_len

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'test': 1, 'val': 1}
        self.set_type = type_map[flag]
        
        self.scale = scale
        self.timeenc = timeenc
        self.flag = flag
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        unlabeled(0), abnormal(-1), normal(1)
        '''
        df_raw['TAG'] = df_raw['TAG'].apply(lambda x: 1 if x=='OK' else (-1 if x=='NG' else 0))
        label = df_raw[['TAG']].values

        '''
        df_raw.columns: ['date', ...(other features)]
        '''
        cols = ['STD_DT', 'MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP']
        cols.remove('STD_DT')
        df_raw = df_raw[['STD_DT'] + cols]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.3)
        border1s = [0, len(df_raw) - num_test]
        border2s = [num_train, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        train_data = data[border1s[0]:border2s[0]]
        train_label = label[border1s[0]:border2s[0]]
        x_train, x_valid, y_train, y_valid = train_test_split(train_data, train_label, test_size=0.3)
        
        if self.flag == 'test':
            self.data_x = data[border1:border2]
            self.data_y = label[border1:border2]
        elif self.flag == 'train':
            self.data_x = x_train
            self.data_y = y_train
        elif self.flag == 'val':
            self.data_x = x_valid 
            self.data_y = y_valid

    def __getitem__(self, index):
        s_begin = index
        
        s_end = s_begin + self.seq_len
        
        # seq_x = self.data_x[s_begin:s_end].reshape(-1, self.seq_len)
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_begin:s_end]
        seq_x_mark = self.data_x[s_begin:s_end] # 사용 X data input 맞춰주기 위해 
        
        return seq_x, seq_y, seq_x_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred2(Dataset):
    def __init__(self, root_path, flag='test', seq_len=50,
                 data_path='competition.csv', scale=True, timeenc=0):
        
        # info 
        self.seq_len = seq_len

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc
        self.stride = seq_len 

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        unlabeled(0), abnormal(-1), normal(1)
        '''
        df_raw['TAG'] = df_raw['TAG'].apply(lambda x: 1 if x=='OK' else (-1 if x=='NG' else 0))
        label = df_raw[['TAG']].values

        '''
        df_raw.columns: ['date', ...(other features)]
        '''
        cols = ['STD_DT', 'MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP']
        cols.remove('STD_DT')
        df_raw = df_raw[['STD_DT'] + cols]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['STD_DT']][border1:border2]
        df_stamp['STD_DT'] = pd.to_datetime(df_stamp.STD_DT)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.STD_DT.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.STD_DT.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.STD_DT.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.STD_DT.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['STD_DT'], 1).values
        elif self.timeenc == 1: # 아무것도 안들어가는 걸로 수정 필요 !
            df_stamp['month'] = df_stamp.STD_DT.apply(lambda row: row.month, 1)
            data_stamp = df_stamp.drop(['STD_DT'], 1).values

        self.data_x = data[border1:border2]
        self.data_y = label[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index 
        
        s_end = s_begin + self.seq_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        return seq_x, seq_y, seq_x_mark, index
    
    def get_total_label(self):
        return self.data_y

    def __len__(self):
        return (len(self.data_x) - self.seq_len) + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    
#-------------------------------------------------------------------------------------#
#-- NAB dataset only -----------------------------------------------------------------#

class Dataset_NAB(Dataset):
    def __init__(self, root_path, flag='train', seq_len=50,
                 data_path='art_daily_jumpsdown.csv', scale=True, timeenc=0):
        
        # info 
        self.seq_len = seq_len

        # init
        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        unlabeled(0), abnormal(-1), normal(1)
        '''
        label = df_raw[['anomaly']].values

        '''
        df_raw.columns: ['date', ...(other features)]
        '''
        cols = ['timestamp', 'value']
        cols.remove('timestamp')
        df_raw = df_raw[['timestamp'] + cols]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.3)
        border1s = [0, num_train]
        border2s = [num_train, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['timestamp']][border1:border2]
        df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['timestamp'], 1).values
        elif self.timeenc == 1: # 아무것도 안들어가는 걸로 수정 필요 !
            df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month, 1)
            data_stamp = df_stamp.drop(['timestamp'], 1).values

        self.data_x = data[border1:border2]
        self.data_y = label[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        
        s_end = s_begin + self.seq_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        return seq_x, seq_y, seq_x_mark 

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
        
        
class Dataset_Pred2_nab(Dataset):
    def __init__(self, root_path, flag='test', seq_len=50,
                 data_path='art_daily_jumpsdown.csv', scale=True, timeenc=0):
        
        # info 
        self.seq_len = seq_len

        # init
        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        unlabeled(0), abnormal(-1), normal(1)
        '''
        label = df_raw[['anomaly']].values

        '''
        df_raw.columns: ['date', ...(other features)]
        '''
        cols = ['timestamp', 'value']
        cols.remove('timestamp')
        df_raw = df_raw[['timestamp'] + cols]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.3)
        border1s = [0, num_train]
        border2s = [num_train, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['timestamp']][border1:border2]
        df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['timestamp'], 1).values
        elif self.timeenc == 1: # 아무것도 안들어가는 걸로 수정 필요 !
            df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month, 1)
            data_stamp = df_stamp.drop(['timestamp'], 1).values

        self.data_x = data[border1:border2]
        self.data_y = label[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index 
        
        s_end = s_begin + self.seq_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        return seq_x, seq_y, seq_x_mark, index
    
    def get_total_label(self):
        return self.data_y

    def __len__(self):
        return (len(self.data_x) - self.seq_len) + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    
#-------------------------------------------------------------------------------------#
#-- ASD dataset only -----------------------------------------------------------------#
class Dataset_ASD(Dataset):
    def __init__(self, root_path, flag='train', seq_len=50,
                 data_path='ASD_omi_1.csv', scale=True, timeenc=0):
        
        # info 
        self.seq_len = seq_len

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        unlabeled(0), abnormal(-1), normal(1)
        '''
        df_raw['TAG'] = df_raw['label'].apply(lambda x: 1 if x==0 else (-1 if x==1 else 0))
        label = df_raw[['TAG']].values
        df_raw.drop('label', axis=1, inplace=True)

        '''
        df_raw.columns: [(other features)]
        '''
        cols = []
        p = re.compile('A[0-9]')
        for col in df_raw.columns:
            if p.match(col):
                cols.append(col)
        df_raw = df_raw[cols]
        
        # print(cols)
        num_train = 8640 - 1000
        num_test = 4320
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = label[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        
        s_end = s_begin + self.seq_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_begin:s_end]
        
        return (seq_x, seq_y) if self.set_type==0 else (seq_x, seq_y, index)
    
    def get_total_label(self):
        return self.data_y

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)