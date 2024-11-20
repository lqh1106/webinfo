import jieba
import pandas as pd
import re
import opencc
import pkuseg

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


def save_data(data, file_path):
    df = pd.DataFrame(data)
    # Save as CSV
    df.to_csv(file_path, index=False, encoding='utf-8')


def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    return set(stopwords)


def is_numeric_or_symbol(word):
    # 匹配纯数字或纯符号的字符串
    return bool(re.fullmatch(r'[\d]+|[^\w\s]+', word))

def get_tags(original_data, original_data_type, stopwords, synonym_dict, cut_type, file_path, pkuseg):
    data = {original_data_type: [], "Tags": []}
    for number, tags in zip(original_data[original_data_type], original_data['Tags']):
        tags = tags.strip("{}")
        tags_list = [item.strip().strip("','")
                        for item in tags.split(",")]
        words = []
        for tag in tags_list:
            simplified_tag = convert_text(tag, 't2s') #繁体转简体
            if cut_type == 'jieba':
                cut_words = [word for word in jieba.lcut(
                    simplified_tag) if not is_numeric_or_symbol(word)] #jieba分词，并删除纯数字或符号串
            elif cut_type == 'pkuseg':
                cut_words = [word for word in pkuseg.cut(
                    simplified_tag) if not is_numeric_or_symbol(word)] #jieba分词，并删除纯数字或符号串
            a = replace_with_center_word(cut_words, synonym_dict) # 同义词替换为中心词
            a = [word for word in a if word not in stopwords] #删除停用词
            words += a
        data[original_data_type].append(number)
        data['Tags'].append(words)
    save_data(data, file_path)

def is_english(text):
    # 判断是否为英文
    return bool(re.match(r'^[a-zA-Z]+$', text))

def bidirectional_maximum_matching(text, dictionary):
    # 正向最大匹配
    def forward_match(text):
        words = []
        while text:
            for i in range(len(text), 0, -1):
                # 将英文单词整个保存，不分词
                if is_english(text[:i]):
                    words.append(text[:i])
                    text = text[i:]
                    break
                if text[:i] in dictionary:
                    words.append(text[:i])
                    text = text[i:]
                    break
            else:
                words.append(text[0])
                text = text[1:]
        return words

    # 反向最大匹配
    def backward_match(text):
        words = []
        while text:
            for i in range(len(text), 0, -1):
                # 将英文单词整个保存，不分词
                if is_english(text[-i:]):
                    words.append(text[-i:])
                    text = text[:-i]
                    break
                if text[-i:] in dictionary:
                    words.append(text[-i:])
                    text = text[:-i]
                    break
            else:
                words.append(text[-1])
                text = text[:-1]
        return words[::-1]

    forward_words = forward_match(text)
    backward_words = backward_match(text)
    
    # 选择较短的分词结果
    if len(forward_words) >= len(backward_words):
        return backward_words
    return forward_words

def get_tags_maxmatch(original_data, original_data_type, dictionary, stopwords, synonym_dict, file_path):
    data = {original_data_type: [], "Tags": []}
    for number, tags in zip(original_data[original_data_type], original_data['Tags']):
        tags = tags.strip("{}")
        tags_list = [item.strip().strip("','")
                        for item in tags.split(",")]
        words = []
        for tag in tags_list:
            simplified_tag = convert_text(tag, 't2s') #繁体转简体
            cut_words = [
                word for word in bidirectional_maximum_matching(simplified_tag, dictionary) if not 
                is_numeric_or_symbol(word)]
            a = replace_with_center_word(cut_words, synonym_dict)# 同义词替换为中心词
            a = [word for word in a if word not in stopwords]#删除停用词
            words += a
        data[original_data_type].append(number)
        data['Tags'].append(words)
    save_data(data, file_path)

if __name__ == '__main__':
    stopwords = load_stopwords('webinfo/data/baidu_stopwords.txt')
    synonym_dict = load_synonym_dict('webinfo/data/dict_synonym.txt')
    original_data_book = pd.read_csv('webinfo/data/selected_book_top_1200_data_tag.csv')
    pkuseg = pkuseg.pkuseg() 
    get_tags(original_data_book, 'Book', stopwords, synonym_dict, 'jieba', 'webinfo/data/book_tag.csv', pkuseg)
