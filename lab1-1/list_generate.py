import pandas as pd
import ast
import json

# 指定文件路径
file_path = 'lab1-1/dataset/book_tag.csv'

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
        
# 对倒排索引表中的文档ID列表进行排序
for tag in inverted_index:
    
    inverted_index[tag].sort()
    
    # pre = 0
    # for index in range(len(inverted_index[tag])):
    #     if index == 0:
    #         pre = inverted_index[tag][index]
    #         continue
    #     minus = inverted_index[tag][index] - pre
    #     pre = inverted_index[tag][index]
    #     inverted_index[tag][index] = minus
        
    inverted_index[tag] = [len(inverted_index[tag]), inverted_index[tag]]# 储存频率和文档ID列表
        
# 将倒排表按tag排序
inverted_index = dict(sorted(inverted_index.items()))
            
            
            
# 将倒排索引字典写入到一个JSON文件中
with open('lab1-1/dataset/inverted_index.json', 'w', encoding='utf-8') as f:
    json.dump(inverted_index, f, ensure_ascii=False, indent=4)

print("倒排索引表已生成并存储到'inverted_index.json'文件中。")
