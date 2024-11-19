import pandas as pd
import ast
import json

def generate_inverted_index(file_path, file_type, output_path):
    """
    从给定的CSV文件生成倒排索引，并将其保存为JSON文件。
    
    :param file_path: 输入的CSV文件路径
    :param output_path: 输出的JSON文件路径
    """
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 初始化倒排索引字典
    inverted_index = {}

    # 遍历每一行数据
    for index, row in data.iterrows():
        book_id = row[file_type]
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
        # 储存频率和文档ID列表
        inverted_index[tag] = [len(inverted_index[tag]), inverted_index[tag]]
        
    # 将倒排表按tag排序
    inverted_index = dict(sorted(inverted_index.items()))

    # 将倒排索引字典写入到一个JSON文件中
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, ensure_ascii=False, indent=4)

    print(f"倒排索引表已生成并存储到'{output_path}'文件中。")


# 使用示例
if __name__ == '__main__':
    # generate_inverted_index('lab1-1/dataset/book_tag_pkuseg.csv', 'Book', 'lab1-1/dataset/book_index.json')
    # generate_inverted_index('lab1-1/dataset/movie_tag_pkuseg.csv', 'Movie', 'lab1-1/dataset/movie_index.json')
    generate_inverted_index('lab1-1/dataset/book_tag_without_synonym_pkuseg.csv', 'Book', 'lab1-1/dataset/book_index_without_synonym.json')
    generate_inverted_index('lab1-1/dataset/book_tag_without_stopwords_pkuseg.csv', 'Book', 'lab1-1/dataset/book_index_without_stopwords.json')    

