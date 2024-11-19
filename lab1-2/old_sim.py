import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import json

def create_id_mapping(id_list):
    id_to_idx = {id_: idx for idx, id_ in enumerate(id_list)}
    idx_to_id = {idx: id_ for id_, idx in id_to_idx.items()}
    return id_to_idx, idx_to_id

def cal_similarity(book_id1, book_id2, tfidf_df):
    vector1 = tfidf_df[tfidf_df['Book'] == book_id1].drop(columns=['Book']).values
    vector2 = tfidf_df[tfidf_df['Book'] == book_id2].drop(columns=['Book']).values
    similarity = cosine_similarity(vector1, vector2)
    return similarity[0][0]

# 读取数据
loaded_data = pd.read_csv('../lab1-1/dataset/book_score.csv')
user_ids = loaded_data['User'].unique()
book_ids = loaded_data['Book'].unique()
file_path = 'lab1-2/data/tfidf_result.csv'
tfidf_df = pd.read_csv(file_path)
# 创建 ID 映射
user_to_idx, idx_to_user = create_id_mapping(user_ids)
book_to_idx, idx_to_book = create_id_mapping(book_ids)

u_items_list = {}
loaded_data['user_map'] = loaded_data['User'].map(user_to_idx)
loaded_data['book_map'] = loaded_data['Book'].map(book_to_idx)

# 按映射后的用户 ID 分组
grouped_user = loaded_data.groupby('user_map')
grouped_book = loaded_data.groupby('book_map')

# 遍历排序后的分组
for user, group in tqdm(grouped_user):
    books = group['book_map'].tolist()
    rates = group['Rate'].tolist()
    
    
    # 预计算用户书籍的相似度矩阵
    book_vectors = tfidf_df[tfidf_df['Book'].isin([idx_to_book.get(book) for book in books])].iloc[:, 1:].values
    similarity_matrix = cosine_similarity(book_vectors)
    if len(books) != similarity_matrix.shape[0] and len(books) != similarity_matrix.shape[1]:
        continue
    
    u_items_list[str(idx_to_user.get(user))] = []
    
    for i in range(len(books)):
        if rates[i] == 0:
            continue
        u_items_list[str(idx_to_user.get(user))].append([str(idx_to_book.get(books[i])), int(rates[i]), []])
        for j in range(len(books)):
            if rates[j] == 0:
                continue
            similarity = similarity_matrix[i][j]
            u_items_list[str(idx_to_user.get(user))][-1][2].append((int(rates[j]), float(similarity)))
            
    # u_items_list[str(idx_to_user.get(user))].append([str(idx_to_book.get(books[0])), int(rates[0]), []])
    # for j in range(len(books)):
    #         if rates[j] == 0:
    #             continue
    #         similarity = similarity_matrix[0][j]
    #         u_items_list[str(idx_to_user.get(user))][-1][2].append((int(rates[j]),str(idx_to_book.get(books[j])), float(similarity)))

# 将字典的键转换为字符串
u_items_list_str_keys = {str(k): v for k, v in u_items_list.items()}

# 保存为 JSON 文件
with open('./data/sim_score_minus.json', 'w') as f:
    json.dump(u_items_list_str_keys, f, indent=4)