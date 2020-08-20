"""
对jester数据集进行预处理和读取。
jester数据集是一个笑话数据集，我们选取了其中48483个用户对100个笑话的评分。
每个用户至少对36个笑话进行了评分，因此我们将每个用户的18个评分划分到测试集中，其余划分到训练集中。
jester数据集中的评分从-10到10，有两位小数。数值99表示用户没有对这个笑话进行评分。
我们在数据处理的时候将评分范围改为了[1,21]，用0表示用户未对笑话进行评分。
"""

import random

import xlrd
import torch
import torch.utils.data as Data


def new_round(_float, _len):
    if isinstance(_float, float):
        if str(_float)[::-1].find('.') <= _len:
            return _float
        if str(_float)[-1] == '5':
            return round(float(str(_float)[:-1] + '6'), _len)
        else:
            return round(_float, _len)
    else:
        return round(_float, _len)


def train_test_split():
    train_writer = open('../jester/train_ratings.csv', 'w')
    test_writer = open('../jester/test_ratings.csv', 'w')

    def f(sheet):
        for i in range(sheet.nrows):
            cells = sheet.row_values(i)
            ratings = []
            non_zero_idx = []
            for j in range(1, 101):
                rate = float(cells[j])
                if rate == 99:
                    ratings.append(0)
                else:
                    # 记录已评分的下标
                    non_zero_idx.append(j - 1)
                    ratings.append(new_round(rate + 11, 2))
            # 随机选取18个作为测试集
            test_idx = random.sample(non_zero_idx, 18)
            train_row = ''
            test_row = ''
            for j in range(100):
                if ratings[j] == 0:
                    train_row = train_row + str(0) + ','
                    test_row = test_row + str(0) + ','
                elif j in test_idx:
                    train_row = train_row + str(0) + ','
                    test_row = test_row + str('{:.2f}'.format(ratings[j])) + ','
                else:
                    train_row = train_row + str('{:.2f}'.format(ratings[j])) + ','
                    test_row = test_row + str(0) + ','
            train_writer.write(train_row[:-1] + '\n')
            test_writer.write(test_row[:-1] + '\n')

    try:
        data = xlrd.open_workbook('../jester/jester-data-1.xls')
        sheet = data.sheet_by_name('jester-data-1-new')
        print('对jester-data-1.xls进行处理...')
        f(sheet)

        data = xlrd.open_workbook('../jester/jester-data-2.xls')
        sheet = data.sheet_by_name('jester-data-2-new')
        print('\n对jester-data-2.xls进行处理...')
        f(sheet)
    finally:
        train_writer.close()
        test_writer.close()


class JesterRatingDataset(Data.Dataset):
    def __init__(self, rating_path, num_users=48483, num_jokes=100):
        self.ratings = torch.zeros(num_users, num_jokes)
        with open(rating_path, 'r') as f:
            cnt = 0
            for line in f:
                rating = line.strip().split(',')
                for i in range(num_jokes):
                    self.ratings[cnt][i] = float(rating[i])
                cnt = cnt + 1

    def __getitem__(self, index):
        return self.ratings[index]

    def __len__(self):
        return len(self.ratings)


if __name__ == '__main__':
    train_test_split()
