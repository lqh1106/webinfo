# import pandas as pd
# import ast
# from collections import Counter

# # 读取三个 CSV 文件的前三行
# df_jieba = pd.read_csv('lab1-1/dataset/book_tag.csv').head(3)
# df_pkuseg = pd.read_csv('lab1-1/dataset/book_tag_pkuseg.csv').head(3)
# df_maxmatch = pd.read_csv('lab1-1/dataset/book_tag_maxmatch.csv').head(3)

# # 将 "Tags" 列的字符串转换为列表
# df_jieba['Tags'] = df_jieba['Tags'].apply(lambda x: ast.literal_eval(x))
# df_pkuseg['Tags'] = df_pkuseg['Tags'].apply(lambda x: ast.literal_eval(x))
# df_maxmatch['Tags'] = df_maxmatch['Tags'].apply(lambda x: ast.literal_eval(x))

# jieba_tags = [tag for tags in df_jieba['Tags'] for tag in tags]
# pkuseg_tags = [tag for tags in df_pkuseg['Tags'] for tag in tags]
# maxmatch_tags = [tag for tags in df_maxmatch['Tags'] for tag in tags]

# # Jaccard 相似度计算
# def jaccard_similarity(A, B):
#     counter_A = Counter(A)
#     counter_B = Counter(B)

#     # 计算交集的大小
#     intersection = sum((counter_A & counter_B).values())
#     # 计算并集的大小
#     union = sum((counter_A | counter_B).values())

#     return intersection / union if union != 0 else 0

# # 计算相似度
# similarity_results = {
#     'jieba-pkuseg': jaccard_similarity(jieba_tags, pkuseg_tags),
#     'jieba-maxmatch': jaccard_similarity(jieba_tags, maxmatch_tags),
#     'pkuseg-maxmatch': jaccard_similarity(pkuseg_tags, maxmatch_tags)
# }

# # 输出结果
# for key, value in similarity_results.items():
#     print(f"{key}: {value:.8f}")

import pandas as pd
import ast
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 读取三个 CSV 文件的前三行
df_jieba = pd.read_csv('lab1-1/dataset/book_tag.csv')
df_pkuseg = pd.read_csv('lab1-1/dataset/book_tag_pkuseg.csv')
df_maxmatch = pd.read_csv('lab1-1/dataset/book_tag_maxmatch.csv')

# 将 "Tags" 列的字符串转换为列表
df_jieba['Tags'] = df_jieba['Tags'].apply(lambda x: ast.literal_eval(x))
df_pkuseg['Tags'] = df_pkuseg['Tags'].apply(lambda x: ast.literal_eval(x))
df_maxmatch['Tags'] = df_maxmatch['Tags'].apply(lambda x: ast.literal_eval(x))

# 将所有标签展开为单个列表
jieba_tags = [tag for tags in df_jieba['Tags'] for tag in tags]
pkuseg_tags = [tag for tags in df_pkuseg['Tags'] for tag in tags]
maxmatch_tags = [tag for tags in df_maxmatch['Tags'] for tag in tags]

# 计算词频向量
def compute_tf(tags):
    counter = Counter(tags)
    total = len(tags)
    return {tag: count / total for tag, count in counter.items()}

jieba_tf = compute_tf(jieba_tags)
pkuseg_tf = compute_tf(pkuseg_tags)
maxmatch_tf = compute_tf(maxmatch_tags)

# 将词频向量转换为向量矩阵
def to_vector(tf, all_tags):
    return [tf.get(tag, 0) for tag in all_tags]

all_tags = sorted(set(jieba_tags + pkuseg_tags + maxmatch_tags))
jieba_vector = to_vector(jieba_tf, all_tags)
pkuseg_vector = to_vector(pkuseg_tf, all_tags)
maxmatch_vector = to_vector(maxmatch_tf, all_tags)

# 计算余弦相似度
vectors = np.array([jieba_vector, pkuseg_vector, maxmatch_vector])
similarity_matrix = cosine_similarity(vectors)

# 输出结果
similarity_results = {
    'jieba-pkuseg': similarity_matrix[0, 1],
    'jieba-maxmatch': similarity_matrix[0, 2],
    'pkuseg-maxmatch': similarity_matrix[1, 2]
}

for key, value in similarity_results.items():
    print(f"{key}: {value:.8f}")
