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
with open('./data/sim_score.json', 'r', encoding='utf-8') as f:
    inverted_index = json.load(f)

# 倒排索引字典记录 user_id -> { { book_id, cosine_similarity, rate}, ...}
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
    with open('./data/Contacts.txt', 'r') as f:
        contacts = f.readlines()
    contacts = [contact.strip().split(':') for contact in contacts]
    contacts = {contact[0]: contact[1].split(',') for contact in contacts}

    # 计算用户的朋友的评分信息
    friends_score = []
    print(user_id)
    for friend in contacts[user_id]:
        friend_score = score[score['User'] == int(friend)]
        if friend_score.empty:
            continue
        friends_score.append(friend_score)

    # 计算用户的朋友的评分信息的倒排索引
    friends_inverted_index = {}
    for friend_score in friends_score:
        for index, row in friend_score.iterrows():
            book_id = row['Book'] if is_book else row['Movie']
            rate = row['Rate']
            if book_id in user_score['Book'].values: # 如果用户已经评分过该书籍，则不考虑
                continue
            if book_id in friends_inverted_index:
                friends_inverted_index[book_id].append(rate)
            else:
                friends_inverted_index[book_id] = [rate]

    # 计算朋友的评分书籍与用户的相似度
    for book_id in friends_inverted_index:
        # 计算 book_id 在user_id评价过的书籍中的相似度
        cosine_similarity = 0
        srate = 0
        for i in inverted_index[user_id][2]:
            if i[0] == book_id:
                cosine_similarity = i[1]
                srate = i[2]
                break
        friends_inverted_index[book_id] = cosine_similarity * srate * np.mean(friends_inverted_index[book_id])
    
    print (friends_inverted_index)
    sorted_friends_inverted_index = sorted(friends_inverted_index.items(), key=lambda x: x[1], reverse=True)
    return sorted_friends_inverted_index[:k]
    

print(knn_recommend('34894527', 5, True))