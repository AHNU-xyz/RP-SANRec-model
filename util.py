import pandas as pd
import numpy as np
import math
import random
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_augmentation import *


def sparseFeature(feat, feat_num, embed_dim=100):
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}

def datasets_preprocessing(file, trans_score=2, embed_dim=100, maxlen=5):
    print('==========数据预处理 Start！============')

    data_df = pd.read_csv(file, sep='\t', engine='python', names=['user_id', 'item_id', 'label', 'Timestamp'])

    data_df = data_df[data_df.label >= trans_score]
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])

    train_data, val_data, test_data = [], [], []

    item_id_max = data_df['item_id'].max() # 1682
    user_id_max = data_df['user_id'].max() # 943
    data_df = data_augmentation()

    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(1, item_id_max)
            return neg
        neg_list = [gen_neg() for i in range(len(pos_list)+100)]

        for i in range(1, len(pos_list)):
            hist_i = pos_list[:i]
            if i == len(pos_list) - 1:
                for neg in neg_list[i:]:
                    test_data.append([user_id, hist_i, pos_list[i], neg])
            elif i == len(pos_list) - 2:
                val_data.append([user_id, hist_i, pos_list[i], neg_list[i]])
            else:
                train_data.append([user_id, hist_i, pos_list[i], neg_list[i]])


    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1
    feature_columns = [sparseFeature('user_id', user_num, embed_dim),
                       sparseFeature('item_id', item_num, embed_dim)]

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    train = pd.DataFrame(train_data, columns=['user_id', 'hist', 'pos_item', 'neg_item'])
    val = pd.DataFrame(val_data, columns=['user_id', 'hist', 'pos_item', 'neg_item'])
    test = pd.DataFrame(test_data, columns=['user_id', 'hist', 'pos_item', 'neg_item'])


    print('=====================================')
    def df_to_list(data):
        return [data['user_id'].values, pad_sequences(data['hist'], maxlen=maxlen),
                data['pos_item'].values, data['neg_item'].values]

    train = df_to_list(train)
    val = df_to_list(val)
    test = df_to_list(test)
    print('============数据预处理阶段 结束！！！=============')
    return feature_columns, train, val, test

# feature_columns, train, val, test = datasets_preprocessing('../dataset/ml-100k/u.data',trans_score=2, embed_dim=100, maxlen=5)

