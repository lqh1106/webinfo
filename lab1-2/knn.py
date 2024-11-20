import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import json
import pandas as pd
from tqdm import tqdm
from knn_recommend import get_fri_sim,get_fri_books

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


def slicing(ratings,similarities,k):
    sliced_ratings = []
    sliced_similarities = []
    for rating,similarity in zip(ratings,similarities):
        if len(rating) < k:
            sliced_ratings.append(rating)
            sliced_similarities.append(similarity)
        else:
            combined = list(zip(rating,similarity))
            combined.sort(reverse=True, key=lambda x: x[1])
            rating,similarity = zip(*combined[:k])
            sliced_ratings.append(rating)
            sliced_similarities.append(similarity)
    return sliced_ratings,sliced_similarities

def get_random_items_from_dict(d, n):
    if n > len(d):
        raise ValueError("n is larger than the number of items in the dictionary")
    dlist = list(d.items())
    dlist = [dl for dl in dlist if len(dl[1]) > 50]
    sample = random.sample(dlist, n)
    return dict(sample)

def knn_sort(k ,lam,FRIEND,sim_score,tfidf_df,contacts):

    total_ndcg = 0
    total_mse = 0
    length = 0
    for key ,item in tqdm(sim_score.items()):
        try:
            if len(item) == 1:
                continue
                
            target_ratings = []
            answer_ratings = []
            ratings = []
            similarities = []
            fri_ratings=[]
            fri_similarities =[]
            if FRIEND:
                fri_rating,ordered_vectors = get_fri_books(key,contacts,tfidf_df)
                
            for j in item:
                if len(j[2])==1:
                    continue

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
                
                if FRIEND:
                    target_id = int(j[0])
                    fri_similarity =get_fri_sim(target_id,tfidf_df,ordered_vectors)
                    fri_ratings.append(fri_rating)
                    fri_similarities.append(fri_similarity)
                    
            if target_ratings == []:
                continue
                
            ratings,similarities = slicing(ratings,similarities,k)
            answer_ratings = [np.dot(r,s)/sum(s) for r,s in zip(ratings,similarities)]
            
            if FRIEND and fri_ratings != []:
                fri_ratings,fri_similarities = slicing(fri_ratings,fri_similarities,k)
                answer_fri_ratings = [np.dot(r,s)/sum(s) for r,s in zip(fri_ratings,fri_similarities)]
                answer_ratings = lam * np.array(answer_ratings) + (1-lam) * np.array(answer_fri_ratings)
            
            answer_sort = sort_by_rate(answer_ratings,target_ratings)
            target_sort = sorted(target_ratings,reverse=True)
            ndcg_score = ndcg(target_sort, answer_sort)
            total_ndcg += ndcg_score[4]
            total_mse += mean_squared_error(target_sort, answer_sort)
            length += 1
        except:
            continue
    print(f'NDCG: {total_ndcg/length}, MSE : {total_mse/length}')
    return total_ndcg/length,total_mse/length

if __name__ == '__main__':
    with open('lab1-2/data/sim_score.json', 'r') as f:
        sim_score = json.load(f)
    sim_score = random_items = get_random_items_from_dict(sim_score, 20)
    file_path = 'lab1-2/data/tfidf_result.csv'
    tfidf_df = pd.read_csv(file_path)
    
    with open('lab1-2/data/Contacts.txt', 'r') as f:
        contacts = f.readlines()
    
    ndcg_lam = []
    mse_lam = []
    lam_range=[i / 10 for i in range(1,11)]
    print("lam_range")
    for lam in tqdm(lam_range):
        ave_ndcg,ave_mse = knn_sort(10,lam,True,sim_score,tfidf_df,contacts)
        ndcg_lam.append(ave_ndcg)
        mse_lam.append(ave_mse)
    print("k_range")
    
    ndcg_k = []
    mse_k = []
    k_range=[5,10,20,30,40,50]
    for k in tqdm(k_range):
        ave_ndcg,ave_mse = knn_sort(k,0.2,True,sim_score,tfidf_df,contacts)
        ndcg_k.append(ave_ndcg)
        mse_k.append(ave_mse)
        
    output = {}
    output['lam_range'] = lam_range
    output['ndcg_lam'] = ndcg_lam
    output['mse_lam'] = mse_lam
    output['k_range'] = k_range
    output['ndcg_k'] = ndcg_k
    output['mse_k'] = mse_k
    
    with open('lab1-2\data\output.json', 'w') as f:
        json.dump(output, f, indent=4)