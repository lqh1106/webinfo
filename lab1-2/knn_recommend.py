# ./stage2/data/Contacts.txt 存储了关系信息，每行用:分隔，第一个是人的id，后面用,分隔的是他的朋友id
# ./dataset/book_score.csv 存储了书籍的评分信息，存储格式为User,Book,Rate,Time,Tag，第二行开始记录信息
# ./dataset/movie_score.csv 存储了书籍的标签信息，存储格式为User,Movie,Rate,Time,Tag，第二行开始记录信息

import pandas as pd
import ast
import json
from sklearn.metrics.pairwise import cosine_similarity
import math
import numpy as np

# 读取CSV文件
book_score = pd.read_csv('lab1-1/dataset/book_score.csv')
movie_score = pd.read_csv('lab1-1/dataset/movie_score.csv')


# 倒排索引字典记录 user_id -> { { book_id, cosine_similarity, rate}, ...}
# 根据 user_id，使用knn算法对书籍/电影进行推荐，输出结果为 {book_id, rate}的有序列表

def get_fri_books(user_id,contacts,tfidf_df):
    score = book_score


    user_score = score[score['User'] == user_id]

    # 读取用户的朋友信息

    contacts = [contact.strip().split(':') for contact in contacts]
    contacts = {contact[0]: contact[1].split(',') for contact in contacts}

    # 计算用户的朋友的评分信息
    friends_score = []
    for friend in contacts[user_id]:
        friend_score = score[score['User'] == int(friend)]
        if friend_score.empty:
            continue
        friends_score.append(friend_score)

    # 计算用户的朋友的评分信息的倒排索引
    friends_inverted_index = {}
    for friend_score in friends_score:
        for index, row in friend_score.iterrows():
            book_id = row['Book']
            rate = row['Rate']
            if book_id in user_score['Book'].values: # 如果用户已经评分过该书籍，则不考虑
                continue
            if book_id in friends_inverted_index:
                friends_inverted_index[book_id].append(rate)
            else:
                friends_inverted_index[book_id] = [rate]
    fri_rating,fri_books = [],[]
    for i, rates in friends_inverted_index.items():
        for r in rates:
            if r == 0:
                continue
            fri_rating.append(r)
            fri_books.append(i)
    filtered_df = tfidf_df[tfidf_df['Book'].isin(fri_books)]
    ordered_vectors = filtered_df.set_index('Book').loc[fri_books].iloc[:, 1:].values
    
    return fri_rating,ordered_vectors
    
def get_fri_sim(target_id,tfidf_df,ordered_vectors):
    # 读取数据

    
    target_id = [target_id]
    filtered_df1 = tfidf_df[tfidf_df['Book'].isin(target_id)]
    book_vector1 = filtered_df1.set_index('Book').loc[target_id].iloc[:, 1:].values
    
    


    fri_similarity = cosine_similarity(book_vector1,ordered_vectors)
    
    
    # for id, rates in friends_inverted_index.items():
        
    #     book_vector1 = tfidf_df[tfidf_df['Book'] == id].iloc[:, 1:].values
    #     book_vector2 = tfidf_df[tfidf_df['Book'] == book_id].iloc[:, 1:].values
    #     sim = cosine_similarity(book_vector1,book_vector2)[0][0]
    #     for i in range(len(rates)):
    #         fri_ratings.append(rates[i])
    #         fri_similarity.append(sim)
    
    return fri_similarity[0]
    
    