import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 读取 CSV 文件
file_path = 'lab1-2/data/book_tag.csv'
data = pd.read_csv(file_path)

# 假设 CSV 文件有两列：'Book' 和 'Tags'
# 将 'Tags' 列中的标签合并为一个字符串
data['Tags'] = data['Tags'].apply(lambda x: ' '.join(eval(x)))

# 初始化 TfidfVectorizer
vectorizer = TfidfVectorizer()

# 对 'Tags' 列进行 TF-IDF 分析
tfidf_matrix = vectorizer.fit_transform(data['Tags'])

# 获取词汇表
feature_names = vectorizer.get_feature_names_out()

# 将 TF-IDF 矩阵转换为 DataFrame，并添加书的 ID 列
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
tfidf_df.insert(0, 'Book', data['Book'])

# 输出文档和词的向量化表达
print("TF-IDF 矩阵：")
print(tfidf_df)

# 保存结果到 CSV 文件
tfidf_df.to_csv('lab1-2/data/tfidf_result.csv', index=False)

