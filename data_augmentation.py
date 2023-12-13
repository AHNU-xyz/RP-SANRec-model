import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity

def data_augmentation():
    data_df = pd.read_csv('../dataset/ml-100k/u.data', sep='\t', engine='python', names=['user_id', 'item_id', 'label', 'Timestamp'])
    data_df = data_df[data_df.label >= 2]
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])

    def get_similar_movies(movie_id, numbers):
        # 读取电影类别数据
        genre_cols = ['genre_unknown', 'genre_action', 'genre_adventure', 'genre_animation', 'genre_childrens',
                      'genre_comedy', 'genre_crime', 'genre_documentary', 'genre_drama', 'genre_fantasy',
                      'genre_film_noir',
                      'genre_horror', 'genre_musical', 'genre_mystery', 'genre_romance', 'genre_sci_fi',
                      'genre_thriller',
                      'genre_war', 'genre_western']
        movies = pd.read_csv('../dataset/ml-100k/u.item', sep='|',
                             names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols,
                             encoding='latin-1')

        movie_genre = movies[genre_cols]

        similarity_matrix = cosine_similarity(movie_genre)

        similarity_df = pd.DataFrame(similarity_matrix, index=movies['movie_id'], columns=movies['movie_id'])

        similar_movies = similarity_df.loc[movie_id].sort_values(ascending=False)[1:numbers + 1].index.tolist()
        return similar_movies

    ave_t, ave_seq_len, media = 0.1631, 100, 60
    ave_list, seq_list = [], []

    # ============================================================================================================================

    # 定义4类
    Scarce_list, Sparse_list, Uneven_list, Normal_list = [], [], [], []

    for index, item in data_df[['user_id', 'item_id', 'Timestamp']].groupby('user_id'):

        time_list_tmp = item['Timestamp'].tolist()
        time_list_tmp1 = []
        for i in time_list_tmp:
            time_list_tmp1.append(np.round(i / 1000 / 60 / 60, 2))
        t = np.round(np.var(time_list_tmp1), 4)
        length = len(item)
        if t > ave_t and length < ave_seq_len:
            Scarce_list.append(item)
        elif t > ave_t and length >= ave_seq_len:
            Sparse_list.append(item)
        elif t <= ave_t and length < ave_seq_len:
            Uneven_list.append(item)
        elif t <= ave_t and length >= ave_seq_len:
            Normal_list.append(item)

    MovieId_arr, MovieIndex_arr, SimMovie_arr = [], [], []
    n_arr = []
    user_id_list = []

    df_list = []

    for i in range(len(Scarce_list)):

        item = Scarce_list[i].reset_index(drop=True)

        max_index = item.diff().idxmax(axis=0, skipna=True)[2]
        MovieIndex_arr.append(max_index)
        MovieId_arr.append(item.item_id.loc[max_index])

        n_arr.append(ave_seq_len - len(Scarce_list[i]))

        user_id_list.append(item['user_id'][0])


        sim_item = get_similar_movies(MovieId_arr[i], n_arr[i])

        sim_item = pd.DataFrame({"user_id": sim_item, "item_id": sim_item, "Timestamp": sim_item, })
        sim_item['user_id'] = user_id_list[i]


        item_insert = pd.concat([item.iloc[:MovieIndex_arr[i]], sim_item, item.iloc[MovieIndex_arr[i]:]])
        df_list.append(item_insert)


    result_df = pd.concat(df_list)

    result_df.reset_index(drop=True, inplace=True)



    def dataframe_list_concat(df_list):
        result_df = pd.concat(df_list)
        result_df.reset_index(drop=True, inplace=True)
        return result_df

    res1 = dataframe_list_concat(Sparse_list)
    res2 = dataframe_list_concat(Uneven_list)
    res3 = dataframe_list_concat(Normal_list)

    res = pd.concat([result_df, res1, res2, res3])

    return  res

