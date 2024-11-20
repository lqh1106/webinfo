import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import json

def create_id_mapping(id_list):
    id_to_idx = {id_: idx for idx, id_ in enumerate(id_list)}
    idx_to_id = {idx: id_ for id_, idx in id_to_idx.items()}
    return id_to_idx, idx_to_id


        
def cal_similarity(books,rates,book,tfidf_df):
    
    if books == []:
        return 
    book = [book]
    book_name1= [idx_to_book.get(book) for book in book]
    filtered_df1 = tfidf_df[tfidf_df['Book'].isin(book_name1)]
    book_vector1 = filtered_df1.set_index('Book').loc[book_name1].iloc[:, 1:].values
    
    
    book_names = [idx_to_book.get(book) for book in books]
    filtered_df = tfidf_df[tfidf_df['Book'].isin(book_names)]
    book_vectors = filtered_df.set_index('Book').loc[book_names].iloc[:, 1:].values
    similarity_matrix = cosine_similarity(book_vector1,book_vectors)

    
    sim_scores = [
    [str(idx_to_book.get(books[j])), similarity_matrix[0][j], rates[j]]
    for j in range(len(books)) if rates[j] != 0
]
    return sim_scores
    

def friends_data():
    contact = {}
    u_u_list = {}
    u_u_i_list = {}
    # 打开文件并读取内容
    with open('webinfo/data/Contacts.txt', 'r') as f:
        for line in f:
            # 分割每一行的内容
            user, friends = line.strip().split(':')
            # 将朋友列表转换为整数列表
            if int(user) in user_to_idx:
                friends_list = [user_to_idx[int(friend)] for friend in friends.split(',') if int(friend) in user_to_idx]
                # 将朋友列表添加到字典中
                contact[user_to_idx[int(user)]] = friends_list

    contact_sorted = {k: v for k, v in sorted(contact.items())}
    u_u_list, u_u_i_list ={},{}
    
    for user, friends in tqdm(contact_sorted.items()):
        u_u_list[user] = friends
        u_u_i_list[user] = []
        for friend in friends:
            u_u_i_list[user] += u_i_list[friend]
                   
    return u_u_list, u_u_i_list 

    
    
if __name__ == '__main__':
    loaded_data = pd.read_csv('webinfo/data/book_score.csv')
    user_ids = loaded_data['User'].unique()
    book_ids = loaded_data['Book'].unique()
    file_path = '/home/gen/lqh/webinfo/data/tfidf_result.csv'
    tfidf_df = pd.read_csv(file_path)
    # 创建 ID 映射
    user_to_idx, idx_to_user = create_id_mapping(user_ids)
    book_to_idx, idx_to_book = create_id_mapping(book_ids)

    u_i_list = {}

    loaded_data['user_map'] = loaded_data['User'].map(user_to_idx)
    loaded_data['book_map'] = loaded_data['Book'].map(book_to_idx)

    # 按映射后的用户 ID 分组
    grouped_user = loaded_data.groupby('user_map')
    grouped_book = loaded_data.groupby('book_map')


    for user, group in tqdm(grouped_user):
        books = group['book_map'].tolist()
        rates = group['Rate'].tolist()
        u_i_list[user] = []
        for book, rate in zip(books, rates):
            if rate > 0:
                u_i_list[user].append([book, rate])
    
    u_i_sim_list = {}
    for user , item in tqdm(u_i_list.items()):
        books = [book[0] for book in item]
        rates = [book[1] for book in item]
        if books == []:
            continue
        sim_score = cal_similarity(books,rates,books[0],tfidf_df)
        if sim_score == None:
            continue
        u_i_sim_list[str(idx_to_user.get(user))] = []
        
        u_i_sim_list[str(idx_to_user.get(user))].append([str(idx_to_book.get(books[0])),rates[0],sim_score])
    with open('webinfo/data/sim_score_minus.json', 'w') as f:
        json.dump(u_i_sim_list, f, indent=4)
            
            
            
    u_u_list, u_u_i_list = friends_data()
    u_u_i_sim_list = {}


    for user , item in tqdm(u_u_i_list.items()):
        try:
            books = [book[0] for book in item]
            rates = [book[1] for book in item] 
            book = u_i_list[user][0][0]
            rate = u_i_list[user][0][1]
            if books == []:
                continue
            sim_score = cal_similarity(books,rates,book,tfidf_df)
            if sim_score == None:
                continue
            u_u_i_sim_list[str(idx_to_user.get(user))] = []
            u_u_i_sim_list[str(idx_to_user.get(user))].append([str(idx_to_user.get(book)),rate,sim_score])
        except:
            continue
    with open('webinfo/data/friend_sim_score_minus.json', 'w') as f:
        json.dump(u_u_i_sim_list, f, indent=4)