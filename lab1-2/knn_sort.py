import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

cosine_similarity_threshold = 0.5
# 读loaded_data取保存的 CSV 文件
loaded_data = pd.read_csv('../lab1-1/dataset/book_score.csv')

# 从 loaded_data 中提取前两列作为 data，第三列作为 tag
data = loaded_data.iloc[:, 0:3].values
tag = loaded_data.iloc[:, 2].values

# 划分训练集和测试集
train_data, test_data, train_tag, test_tag = train_test_split(data, tag, test_size=0.5, random_state=42)

# print(train_data)

# 倒排索引字典 ./data/sim_score_minus 记录 user_id : [ book_id, rate, [ [cosine_similarity ,book_id, rate], ...] ]
# 根据 user_id和索引字典中评价过的书目的cosine_similarity，使用 knn 算法对 test_data 的 rate 进行预测，输出结果为 {book_id, rate}的有序列表
sim_score = json.load(open('./data/sim_score_minus.json', 'r'))

def knn_rate(user_id, book_id, k):
    # 读取用户评分过的书籍
    # print(user_id, book_id)
    rated_books = sim_score[user_id]
    for book in rated_books:
        if book[0] == book_id:
            rated_books = book[2]
            break
    # 根据 rated_books 中的 cosine_similarity 对 book_id 进行排序, 选择前 k 个
    rated_books.sort(key=lambda x: x[2], reverse=True)
    rated_books = rated_books[:k]
    # 计算预测评分
    result = 0
    sum = 0
    for book in rated_books:
        print(book)
        if book[2] > cosine_similarity_threshold:   # 评分大于阈值的才计入
            result += book[0]
            sum += 1
    result /= sum
    return result

if __name__ == "__main__":
    for row in test_data:
        user_id = str(row[0])
        book_id = str(row[1])
        rate = row[2]
        predict = knn_rate(user_id, book_id, 5)
        print(f"User {user_id}, Book {book_id}, predict rate: {predict}, real rate: {rate}")


