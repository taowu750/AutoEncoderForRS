import torch
import torch.utils.data as Data


def count_rating_per_user():
    rating_per_user = {}
    with open('../ml-1m/ratings.dat', 'r') as f:
        for line in f:
            rating = line.strip().split('::')
            uid = int(rating[0])
            if uid in rating_per_user:
                rating_per_user[uid] += 1
            else:
                rating_per_user[uid] = 1

    return rating_per_user


def train_test_split():
    """
    划分训练集、测试集。其中测试集中每个用户有10个评分数据。
    原数据集中每个用户至少有20个评分数据。
    :return:
    """
    temp_uid = 0
    test_counter = 0
    write_test_samples = False

    train_writer = open('../ml-1m/train_ratings.dat', 'w')
    test_writer = open('../ml-1m/test_ratings.dat', 'w')

    with open('../ml-1m/ratings.dat', 'r') as f:
        for line in f:
            rating = line.strip().split('::')
            uid = int(rating[0])

            if temp_uid != uid:
                write_test_samples = True
                temp_uid = uid

            if write_test_samples:
                test_writer.write(line)
                test_counter += 1
                if test_counter >= 10:
                    test_counter = 0
                    write_test_samples = False
            else:
                train_writer.write(line)
        train_writer.close()
        test_writer.close()


def get_ml1m_movie_map():
    movie_title_idx2t = []
    movie_idx2id = []
    movie_id2idx = {}
    with open('../ml-1m/movies.dat', 'r', encoding='utf8') as f:
        movies = f.readlines()
        for i, movie in enumerate(movies):
            movie = movie.strip().split('::')
            movie_idx2id.append(int(movie[0]))
            movie_id2idx[int(movie[0])] = i
            movie_title_idx2t.append(movie[1])

    return movie_id2idx, movie_idx2id, movie_title_idx2t


class Ml1mRatingDataset(Data.Dataset):
    def __init__(self, rating_path, movie_id2idx, num_users=6040, num_movies=3883, *,
                 add_uid=False):
        if add_uid:
            self.ratings = torch.zeros(num_users, num_movies + 1)
        else:
            self.ratings = torch.zeros(num_users, num_movies)
        with open(rating_path, 'r') as f:
            for line in f:
                rating = line.strip().split('::')
                uid = int(rating[0])
                mid = int(rating[1])
                score = int(rating[2])
                if add_uid:
                    self.ratings[uid - 1][0] = add_uid
                    self.ratings[uid - 1][movie_id2idx[mid] + 1] = score
                else:
                    self.ratings[uid - 1][movie_id2idx[mid]] = score

    def __getitem__(self, index):
        return self.ratings[index]

    def __len__(self):
        return len(self.ratings)
