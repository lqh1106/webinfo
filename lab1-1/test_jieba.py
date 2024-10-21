import jieba
import pandas as pd
import re


def load_synonym_dict(file_path):
    synonym_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 使用正则表达式匹配 '=', '#' 或 '@' 作为分隔符
            code, synonyms = re.split(r'[=#@]\s', line.strip(), maxsplit=1)
            synonym_list = synonyms.split(' ')
            if synonym_list:  # 确保有近义词
                center_word = synonym_list[0]  # 第一个词作为中心词
                for word in synonym_list:
                    synonym_dict[word] = center_word
    return synonym_dict


def replace_with_center_word(words, synonym_dict):
    new_words = [synonym_dict.get(word, word) for word in words]
    return new_words


def save_data(data):
    df = pd.DataFrame(data)
    # Save as CSV
    file_path = 'webinfo/lab1-1/dataset/book_tag.csv'
    df.to_csv(file_path, index=False, encoding='utf-8')


def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    return set(stopwords)


# 加载百度停用词库（替换为你的停用词文件路径）
stopwords = load_stopwords('webinfo/lab1-1/dataset/baidu_stopwords.txt')
synonym_dict = load_synonym_dict('webinfo/lab1-1/dataset/dict_synonym.txt')
# 读取CSV文件
file_path = 'webinfo/lab1-1/dataset/selected_book_top_1200_data_tag.csv'  # 替换为你的CSV文件路径
oringal_data = pd.read_csv(file_path)
data = {"Book": [], "Tags": []}
for book, booktags in zip(oringal_data['Book'], oringal_data['Tags']):
    booktags = booktags.strip("{}")
    booktags_list = [item.strip().strip("','")
                     for item in booktags.split(",")]
    words = []
    for tag in booktags_list:
        jieba_words = jieba.lcut(tag)
        filtered_words = [
            word for word in jieba_words if word not in stopwords]
        synonym_replaceed = replace_with_center_word(
            filtered_words, synonym_dict)
        words += synonym_replaceed
    data['Book'].append(book)
    data['Tags'].append(words)
save_data(data)
