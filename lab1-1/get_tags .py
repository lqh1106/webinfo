import jieba
import pandas as pd
import re
import opencc

converter_s2t = opencc.OpenCC('s2t')
converter_t2s = opencc.OpenCC('t2s')


def convert_text(text, conversion_type='s2t'):
    if conversion_type == 's2t':
        return converter_s2t.convert(text)
    elif conversion_type == 't2s':
        return converter_t2s.convert(text)
    else:
        raise ValueError(
            "Unsupported conversion type. Use 's2t' for Simplified to Traditional or 't2s' for Traditional to Simplified.")
        
def load_synonym_dict(file_path):
    synonym_dict = {}
    code = ''
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 使用正则表达式匹配 '=', '#' 或 '@' 作为分隔符
            newcode, synonyms = re.split(r'[=#@]\s', line.strip(), maxsplit=1)
            synonym_list = synonyms.split(' ')
            if synonym_list:  # 确保有近义词
                if newcode[:7] != code[0:7]:
                    code = newcode
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
    file_path = 'lab1-1/dataset/movie_tag.csv'
    df.to_csv(file_path, index=False, encoding='utf-8')


def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    return set(stopwords)


def is_numeric_or_symbol(word):
    # 匹配纯数字或纯符号的字符串
    return bool(re.fullmatch(r'[\d]+|[^\w\s]+', word))


# 加载百度停用词库（替换为你的停用词文件路径）
stopwords = load_stopwords('lab1-1/dataset/baidu_stopwords.txt')
synonym_dict = load_synonym_dict('lab1-1/dataset/dict_synonym.txt')
oringal_data = pd.read_csv('lab1-1/dataset/selected_movie_top_1200_data_tag.csv')


data = {"Movie": [], "Tags": []}
for number, tags in zip(oringal_data['Movie'], oringal_data['Tags']):
    tags = tags.strip("{}")
    tags_list = [item.strip().strip("','")
                     for item in tags.split(",")]
    words = []
    for tag in tags_list:
        jieba_words = [word for word in jieba.lcut(
            tag) if not is_numeric_or_symbol(word)]#jieba分词，并删除纯数字或符号串
        simplified_words=[convert_text(traditional_text, 't2s') for traditional_text in jieba_words]#繁体转简体
        filtered_words = [
            word for word in simplified_words if word not in stopwords]#删除停用词
        synonym_replaceed = replace_with_center_word(
            filtered_words, synonym_dict)# 同义词替换为中心词
        words += synonym_replaceed
        
    data['Movie'].append(number)
    data['Tags'].append(words)
save_data(data)
