# ./stage2/data/Contacts.txt 存储了关系信息，每行用:分隔，第一个是人的id，后面用,分隔的是他的朋友id
# ./dataset/book_score.csv 存储了书籍的评分信息，存储格式为User,Book,Rate,Time,Tag，第二行开始记录信息
# ./dataset/movie_score.csv 存储了书籍的标签信息，存储格式为User,Movie,Rate,Time,Tag，第二行开始记录信息

import pandas as pd
import ast
import json
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# 读取CSV文件
book_score = pd.read_csv('../lab1-1/dataset/book_score.csv')
movie_score = pd.read_csv('../lab1-1/dataset/movie_score.csv')

# 读取倒排索引字典
with open('../lab1-1/dataset/inverted_index.json', 'r') as f:
    inverted_index = json.load(f)

# 倒排索引字典记录 user_id -> { {cosine_similarity, book_id, rate}, ...}
# 根据 user_id，使用knn算法对书籍/电影进行推荐，输出结果为 {book_id, rate}的有序列表
def knn_recommend(user_id, k, is_book):
    # 读取数据
    if is_book:
        score = book_score
    else:
        score = movie_score

    # 读取用户评分信息
    user_score = score[score['User'] == user_id]

    # 读取用户的朋友信息
    with open('./stage2/data/Contacts.txt', 'r') as f:
        contacts = f.readlines()
    contacts = [contact.strip().split(':') for contact in contacts]
    contacts = {contact[0]: contact[1].split(',') for contact in contacts}

    # 计算用户的朋友的评分信息
    friends_score = []
    for friend in contacts[user_id]:
        friend_score = score[score['User'] == int(friend)]
        friends_score.append(friend_score)

    # 计算用户的朋友的评分信息的倒排索引
    friends_inverted_index = {}
    for friend_score in friends_score:
        for index, row in friend_score.iterrows():
            book_id = row['Book'] if is_book else row['Movie']
            rate = row['Rate']
            if book_id in friends_inverted_index:
                friends_inverted_index[book_id].append(rate)
            else:
                friends_inverted_index[book_id] = [rate]

    # 计算用户的朋友的评分信息的平均值
    friends_avg_score = {}
    for book_id, rates in friends_inverted_index.items():
        friends_avg_score[book_id] = sum(rates) / len(rates)

    # 计算用户的朋友的评分信息的标准差
    friends_std_score = {}
    for book_id, rates in friends_inverted_index.items():
        friends_std_score[book_id] = np.std(rates)

    # 计算用户的朋友的评分信息的归一化值
    friends_norm_score = {}
    for book_id, rate in friends_avg_score.items():
        friends_norm_score[book_id] = (rate - min(friends_avg_score.values())) / (max(friends_avg_score.values()) - min(friends_avg_score.values()))

    # 计算用户的朋友的评分信息的相似度
    friends_similarity = {}
    for book_id, rate in friends_norm_score.items():
        friends_similarity[book_id] = 1 - math.sqrt((rate - friends_norm_score[user_id]) ** 2)

    # 计算用户的朋友的评分信息的加权平均值
    friends_weighted_score = {}
    for book_id, rate in friends_avg_score.items():
        friends_weighted_score[book_id] = rate * friends_similarity[book_id]

    # 计算用户的朋友的评分信息的加权平均值的排序
    friends_weighted_score = sorted(friends_weighted_score.items(), key=lambda x: x[1], reverse=True)

    # 计算用户的朋友的评分信息的加权平均值的前k个
    friends_weighted_score = friends_weighted_score[:k]

    # 计算用户的朋友的评分信息的加权平均值的前k个的推荐结果
    friends_recommend = []
    for book_id, rate in friends_weighted_score:
        friends_recommend.append(book_id)

    return friends_recommend