import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import json
import random
from tqdm import tqdm

def sort_by_rate(answer_ratings,target_ratings):


    sorted_indices = np.argsort(answer_ratings)[::-1]
    sorted_ratings = [target_ratings[i] for i in sorted_indices]
    
    return sorted_ratings

def dcg(scores):
    return np.sum([(2**score - 1) / np.log2(idx + 2) for idx, score in enumerate(scores)])

def ndcg(target_sort, answer_sort):
    ndcg = []
    for i in range(len(target_sort)):
        dcg_val = dcg(answer_sort[:i+1])
        idcg_val = dcg(target_sort[:i+1])
        ndcg.append(dcg_val / idcg_val if idcg_val > 0 else 0)
    return ndcg

with open('./data/sim_score.json', 'r') as f:
    data = json.load(f)
    


total_ndcg = 0
total_len = len(data)
for key ,item in tqdm(data.items()):
    try:
        if len(item) == 1:
            continue
        target_ratings = []
        answer_ratings = []
        ratings = []
        similarities = []
        id = []
        for j in item:
            if len(j[2])==1:
                continue
            id.append(j[0])
            target_ratings.append(j[1])
            rating = []
            similaritie = []
            for i in j[2]:
                if i[1] > 0.99:
                    continue
                rating.append(i[0])
                similaritie.append(i[1])
            ratings.append(rating)
            similarities.append(similaritie)
        if target_ratings == []:
            continue    
        
        answer_ratings = [np.dot(r,s)/sum(s) for r,s in zip(ratings,similarities)]
        
        answer_sort = sort_by_rate(answer_ratings,target_ratings)
        target_sort = sorted(target_ratings,reverse=True)
        ndcg_score = ndcg(target_sort, answer_sort)
        
        total_ndcg += ndcg_score[-1]
        
    except:
        total_len -= 1
        continue
    
print(f'NDCG Score: {total_ndcg/total_len}')