"""
数据集 ml1m 的加载模块
"""


import numpy as np
import torch
import torch.utils.data as Data


# 18 种电影类型
movie_genres_i2t = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movie_genres_t2i = {token: idx for idx, token in enumerate(movie_genres_i2t)}


def get_ml1m_movie_data(return_features=True):
    movie_title_i2t = []
    # 此数据集中的电影数量小于最大 id，中间有一些电影缺失，所以要用到下面的数据结构
    movie_idx2id = []
    movie_id2idx = {}
    with open('../ml-1m/movies.dat', 'r', encoding='utf8') as f:
        movies = f.readlines()
        movie_features = None
        if return_features:
            movie_features = torch.zeros(len(movies), len(movie_genres_i2t), dtype=torch.int8)
        for i, movie in enumerate(movies):
            movie = movie.strip().split('::')
            movie_idx2id.append(int(movie[0]))
            movie_id2idx[int(movie[0])] = i
            movie_title_i2t.append(movie[1])
            if return_features:
                # 将电影类型特征进行 one-hot 编码
                movie_genres = movie[2].split('|')
                movie_genres = [movie_genres_t2i[token] for token in movie_genres]
                movie_features[i][movie_genres] = 1

    if return_features:
        return movie_features, movie_title_i2t, movie_idx2id, movie_id2idx
    else:
        return movie_title_i2t, movie_idx2id, movie_id2idx


def get_ml1m_user_features(feature_type=0):
    """

    :param feature_type: 返回三种类型的 user_features：0 - 只有 id 的 one-hot 编码；1 - id 和其他
    特征的 one-hot 编码；2 - id 为数字；3 - 没有 id 类特征
    :return:
    """

    assert (0 <= feature_type <= 3) and isinstance(feature_type, int)

    # id、性别、年龄、职业
    # 年龄：7个组；职业：21种（0-20）
    age2group = {'1': 0, '18': 1, '25': 2, '35': 3, '45': 4, '50': 5, '56': 6}
    with open('../ml-1m/users.dat', 'r') as f:
        users = f.readlines()
        num_users = len(users)
        if feature_type == 0:
            user_features = torch.zeros(len(users), num_users, dtype=torch.int8)
        elif feature_type == 1:
            user_features = torch.zeros(len(users), num_users + 29, dtype=torch.int8)
        elif feature_type == 2:
            user_features = torch.zeros(len(users), 30, dtype=torch.int16)
        else:
            user_features = torch.zeros(len(users), 29, dtype=torch.int16)
        for i, user in enumerate(users):
            user = user.strip().split('::')
            if feature_type == 0:
                # 将 id 进行 one-hot 编码
                user_features[i][int(user[0]) - 1] = 1
            elif feature_type == 1:
                # 将 id 进行 one-hot 编码
                user_features[i][int(user[0]) - 1] = 1
                user_features[i][num_users] = user[1] == 'M'
                # 将年龄由数值编码变为 one-hot 编码
                user_features[i][num_users + 1 + age2group[user[2]]] = 1
                user_features[i][num_users + 8 + int(user[3])] = 1
            elif feature_type == 2:
                user_features[i][0] = int(user[0])
                user_features[i][1] = user[1] == 'M'
                user_features[i][2 + age2group[user[2]]] = 1
                user_features[i][9 + int(user[3])] = 1
            else:
                user_features[i][0] = user[1] == 'M'
                user_features[i][1 + age2group[user[2]]] = 1
                user_features[i][8 + int(user[3])] = 1

    return user_features


def get_ml1m_ratings():
    with open('../ml-1m/ratings.dat', 'r') as f:
        rating_str = f.readlines()
        ratings = np.zeros((len(rating_str), 3), dtype=np.int16)
        for i, rating in enumerate(rating_str):
            rating = rating.strip().split('::')
            ratings[i][0] = int(rating[0])
            ratings[i][1] = int(rating[1])
            ratings[i][2] = int(rating[2])
        np.random.shuffle(ratings)
        ratings = torch.from_numpy(ratings)

    return ratings


class Ml1mItemDataset(Data.Dataset):
    def __init__(self, ratings, user_features, movie_features, movie_id2idx):
        self.ratings = ratings
        self.user_features = user_features
        self.movie_features = movie_features
        self.movie_id2idx = movie_id2idx

        # 预先分配需要占用很多内存，但是比临时分配要快个 1.6 倍
        # self.user_movies = torch.empty(len(ratings),
        #                                user_features.shape[1] + movie_features.shape[1])
        # for i, rating in enumerate(ratings):
        #     self.user_movies[i][:user_features.shape[1]] = user_features[rating[0].item() - 1]
        #     self.user_movies[i][user_features.shape[1]:] = movie_features[movie_id2idx[rating[1].item()]]

    def __getitem__(self, index):
        rating = self.ratings[index]
        # 需要是 float 类型数据，这样和模型训练时才不会报错。这里默认是 float 数据
        item = torch.empty(self.user_features.shape[1] + self.movie_features.shape[1])
        item[:self.user_features.shape[1]] = self.user_features[rating[0].item() - 1]
        item[self.user_features.shape[1]:] = self.movie_features[self.movie_id2idx[
            rating[1].item()]]
        return item, rating[2]

    def __len__(self):
        return self.ratings.shape[0]
