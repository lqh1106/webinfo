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
    
    
with open('./data/sim_score_minus.json', 'r') as f:
    data = json.load(f)
    
target_ratings = []
ratings = []
similarities = []
id = []


for key ,item in data.items():
    for j in item:
        if len(j[2])==1:
            continue
        id.append(j[0])
        target_ratings.append(j[1])
        rating = []
        similaritie = []
        for i in j[2]:
            if i[0] == id[-1]:
                continue
            rating.append(i[2])
            similaritie.append(i[1])
        ratings.append(rating)
        similarities.append(similaritie)
        

    
max_len = max(len(r) for r in ratings)
ratings_padded = np.array([r + [int(0)] * (max_len - len(r)) for r in ratings])
similarities_padded = np.array([s + [int(0)] * (max_len - len(s)) for s in similarities])

features = np.hstack(similarities_padded)

features = [np.dot(r, s) / sum(s) for r, s in zip(ratings_padded, similarities_padded)]
features = np.array(features).reshape(-1, 1)
target_ratings = np.array(target_ratings)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target_ratings, test_size=0.2, random_state=42)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

with open('./data/sim_score.json', 'r') as f:
    data_test = json.load(f)
    
data = dict(random.sample(data_test.items(), 1000))


total_ndcg = 0
for key ,item in tqdm(data.items()):
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

    ratings_padded = np.array([r + [int(0)] * (max_len - len(r)) for r in ratings])
    similarities_padded = np.array([s + [int(0)] * (max_len - len(s)) for s in similarities])
    new_features = [np.dot(r, s) / sum(s) for r, s in zip(ratings_padded, similarities_padded)]
    new_features = np.array(new_features).reshape(-1, 1)
    answer_ratings = model.predict(np.array(new_features))
    
    answer_sort = sort_by_rate(answer_ratings,target_ratings)
    target_sort = sorted(target_ratings,reverse=True)
    ndcg_score = ndcg(target_sort, answer_sort)
    total_ndcg += ndcg_score[-1]
    

print(f'NDCG Score: {total_ndcg/len(data)}')

