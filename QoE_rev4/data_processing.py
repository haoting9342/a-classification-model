import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math
import joblib
import logging
import pickle
import os
import pdb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import pathlib
import glob
import datetime as dt
import sys
from numpy import newaxis


class DataLoader():

    def __init__(self):
        self.logger = logging.getLogger('QoE.dp')

    def run(self, data_path, label_mapping, group_size=None, save_data=None, normalization=True):
        '''
        Data processing for random forest method
        :param data_path:
        :param label_mapping:
        :param group_size:
        :return:
        '''
        file_list = glob.glob(data_path + "/*.csv")
        df_list = []
        for _file in file_list:
            df = self.data_clean(_file)

            if normalization:
                df = self.data_norm(df)

            if group_size is not None:
                self.logger.info(f"Start multiple sfn grouping, groupsize={group_size} ")
                df = self.data_groupby(df, group_size)

            _file_path, _filename = os.path.split(_file)

            if _filename.startswith('good'):
                label = label_mapping['good']
            elif _filename.startswith('bad'):
                label = label_mapping['bad']
            elif _filename.startswith('normal'):
                label = label_mapping['normal']
            else:
                self.logger.error("The input label is not correct!!!!")
                sys.exit(1)
            df['label'] = label

            df_list.append(df)
        data_all = pd.concat(df_list, axis=0)

        if save_data is not None:
            data_saved_path = os.path.join(save_data, 'processed_data.csv')
            data_all.to_csv(data_saved_path)

        if 'aiSfn' in data_all.columns:
            data_all = data_all.drop('aiSfn', axis=1)
        data_all = data_all.drop(['aiSendCount', 'aiSlot'], axis=1)

        self.logger.info(f"Total number of samples: {data_all.shape}")
        self.logger.info(f"Columns: {data_all.columns}")
        return data_all

    def data_clean(self, filepath, droprows=None):
        # remove unvalid data, delete duplicate samples, sort the samples based on counter
        self.logger.info(f"Data cleaning {filepath}.")
        df = pd.read_csv(filepath, index_col=0, low_memory=False, error_bad_lines=False)
        if droprows is not None:
            df.drop(range(droprows), inplace=True)
            df = df.reset_index(drop=True)
        self.logger.info(f"The shape of data: {df.shape}")
        df = df.drop(df[df['aiIsValid'] == 0].index)
        self.logger.info(f"Number of valid samples: {df.shape}")
        df = self.ordering(df)

        df['time'] = df['aiTimeH'] + df['aiTimeL'] / 1000000
        tdx = df.index[0]
        df['time'] = df['time'] - df['time'][tdx]
        df = df.drop(['aiTimeH', 'aiTimeL'], axis=1)
        self.logger.info(f"Total time duration for this dataset: {df['time'].max()}")

        t_start = time.time()
        df = self.check_duplicate(df)
        t_end = time.time()
        self.logger.info(f"Time for remove duplicate: {t_end - t_start}.")

        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        df = self.shift_sfn(df)

        self.logger.info(f"sfn_max = {df['aiSfn'].max()}, sfn min = {df['aiSfn'].min()}")

        sfn_range = df['aiSfn'].max() - df['aiSfn'].min() + 1
        missing_ratio = 1 - len(np.unique(df['aiSfn'].values)) / sfn_range
        self.logger.info(f"The SFN missing ratio is {missing_ratio}")
        df = df.reset_index(drop=True)

        return df

    def repl_data_type(self, dataset):
        for col in dataset.columns:
            dataset[col] = dataset[col].astype(float)
        return dataset

    def ordering(self, data):
        # ordering based on SendCount, Sfn, and Slot.
        data = data.sort_values(by=['aiSendCount', 'aiSfn', 'aiSlot'], ascending=True)
        data = self.repl_data_type(data)
        further_order_index = data[data['aiSfn'] == 0].index
        for _index in further_order_index:
            count = data.loc[_index, :]['aiSendCount']
            temp = data[data['aiSendCount'] == count]
            # print(temp)
            sfn_range = temp['aiSfn'].unique()
            if 0 in sfn_range and 1023 in sfn_range:
                #self.logger.info(f"index {_index} need to be reordered!!")
                temp = temp.sort_values(by=['aiSfn', 'aiSlot'], ascending=[False, True])
                data[data['aiSendCount'] == count] = temp.values
            # else:
            #    print(f"sfn_range = {sfn_range}")

        return data

        # shifting sfn

    def shift_sfn(self, df):  # input data are numpy
        data = df['aiSfn'].values
        superframe = 0
        firstframe = data[0]
        for i in range(data.shape[0]):
            if i > 0:
                if data[i - 1] - superframe * 1024 > data[i]:
                    superframe += 1
                data[i] += superframe * 1024
        df['aiSfn'] = data - firstframe
        return df

    def check_duplicate(self, df):
        df['delta_ct'] = df['aiSendCount'] // 2000
        dul_index = df.duplicated(subset=['aiSfn', 'aiSlot', 'delta_ct'])
        self.logger.info(df[dul_index].index)
        # df = df[df.duplicated(subset=['aiSfn', 'aiSlot', 'delta_ct'])]
        df = df[~dul_index]
        df.drop(['delta_ct'], axis=1, inplace=True)
        self.logger.info(f"Data shape after duplicate removal: {df.shape}")
        return df

    def data_groupby(self, df, group_size):
        df = self.groupby_multp_sfn(df, group_size)
        #df['Tbs_per_slot'] = df['Tbs']/df['nSlot'] * 100
        self.logger.info(f"Number of samples: {df.shape} ")
        df['aiSfn'] = df['aiSfn']//group_size
        self.logger.info(f"Number of missing samples: {df['aiSfn'].max() + 1 - df.shape[0]}")
        #gp = int(1024/group_size)
        #df['Sfn'] = (df['Sfn'] % gp)/100
        #df['Slot'] = df['Slot']/100
        return df

    def data_norm(self, data):
        # shifting bler, ack, and nack
        temp = data[['aiAck', 'aiNAck']] - data[['aiAck', 'aiNAck']].shift(1)
        temp = temp.fillna(0)
        data[['aiAck', 'aiNAck']] = temp.values
        # insert a column to indicate how many slots within one sfn.
        nSlot = [1 if x > 0 else 0 for x in data['aiSlot']]
        data.insert(4, 'aiSlot_flag', nSlot)
        # data normalization
        data['aiEth'] = data['aiEth'] * 8 / 1024 / 1024  # change unit from Bytes/s to MBits/s
        data['aiTbs'] = data['aiTbs'] * 8 / 1024 / 1024  # unit: Mbps
        data['aiRlcBuff'] = data['aiRlcBuff'] * 8 / 1024 / 1024  # unit: Mbps
        data['aiSendCount'] = data['aiSendCount'] - data['aiSendCount'][0]
        data['aiS1TeID'] = data['aiS1TeID'] - data['aiS1TeID'][0]
        data['aiBler'] = data['aiBler'] / 100 /100  # normalized to 1
        data['aiSinr'] = data['aiSinr'] / 100
        data['aiRMCqi'] = data['aiRMCqi'] / 100
        # data['aiCqiCmp'] = data['aiCqiCmp'] / data['aiCqiCmp'].max()
        feature_set = ['aiSfn', 'aiSlot_flag', 'aiMcs', 'aiTbs', 'aiCqi', 'aiBler', 'aiSinr',
                       'aiAck', 'aiNAck', 'aiLoQueueTime', 'aiEth', 'aiRlcBuff',
                       'aiMod', 'aiRMCqi', 'aiCqiCmp']

        #id_cols = ['aiSendCount', 'time', 'aiSlot', 'aiIsValid', 'aiUeId', 'aiS1TeID', 'aiBearId']
        #unique_cols = ['aiNumOfPrb', 'aiRssi', 'aiTa', 'aiPdcpBuffH', 'aiCqiDel', 'aiTm']
        # unique_cols = ['aiNumOfPrb', 'aiRssi', 'aiSinr', 'aiTa']
        #self.logger.info(f"Delete id cols: {id_cols}.")
        #self.logger.info(f"Delete unique value cols: {unique_cols}")
        #data = data.drop(id_cols, axis=1)
        #data = data.drop(unique_cols, axis=1)
        self.logger.info(f"Selected features set: {feature_set}")
        data = data.loc[:, feature_set]
        data = data.reset_index(drop=True)
        self.logger.info(f"The shape of dataset: {data.shape}")
        return data


    def comb_sfn(self, data, idx, groupsize, pt=False):
        if pt:
            if idx % 100 == 0:
                print(idx)
        return data.loc[idx, 'aiSfn'] // groupsize


    def groupby_multp_sfn(self, data, groupsize):
        print(f"Groupy based on aiSfn with groupsize = {groupsize}...")

        agg_rule = {'aiSfn': np.min,
                    'aiSlot_flag': 'sum',
                    'aiMcs': ['mean', np.min, np.max],
                    'aiCqi': 'mean',
                    'aiBler': 'mean',
                    'aiSinr': 'mean',
                    'aiAck': 'sum',
                    'aiNAck': 'sum',
                    'aiLoQueueTime': 'mean',
                    'aiEth': 'sum',
                    'aiRlcBuff': 'sum',
                    'aiMod': 'mean',
                    'aiRMCqi': 'mean',
                    'aiCqiCmp': 'mean',
                    'aiTbs': 'sum'
                    }
        tmp_dataset = data.groupby(lambda x: self.comb_sfn(data, x, groupsize)).agg(agg_rule)  # per sfn groupby

        tmp_dataset.columns = ['aiSfn', 'nSlot',
                               'Mcs_mean', 'Mcs_min', 'Mcs_max',
                               'Cqi', 'Bler', 'Sinr', 'Ack', 'Nack', 'LoQueueTime', 'Eth',
                               'RlcBuff', 'Modulation', 'RMCqi', 'CqiCmp', 'Tbs']

        tmp_dataset = tmp_dataset.reset_index(drop=True)

        return tmp_dataset