import pandas as pd
import ast
import json

# 指定文件路径
file_path = 'dataset/book_tag.csv'

# 读取CSV文件
data = pd.read_csv(file_path)

# 初始化倒排索引字典
inverted_index = {}

# 遍历每一行数据
for index, row in data.iterrows():
    book_id = row['Book']
    # 将字符串形式的列表转换为真实的Python列表
    tags = ast.literal_eval(row['Tags'])
    
    # 遍历每个标签
    for tag in set(tags):  # 使用set去重，避免重复
        if tag not in inverted_index:
            inverted_index[tag] = []
        inverted_index[tag].append(book_id)

# 将倒排索引字典写入到一个JSON文件中
with open('inverted_index.json', 'w', encoding='utf-8') as f:
    json.dump(inverted_index, f, ensure_ascii=False, indent=4)

print("倒排索引表已生成并存储到'inverted_index.json'文件中。")
