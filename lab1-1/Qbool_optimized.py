from itertools import product
import re
import json
import timeit

# 读取倒排索引
def load_inverted_index(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

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
    # 去掉重复的词
    new_words = list(set(new_words))
    return new_words

def extract_variables(query_string, synonym_dict):
    # 使用正则表达式提取变量, 并替换同义词
    pattern = r'\b(?!AND|OR|NOT)(\w+)\b'  # 匹配字母组成的单词
    all_variables = re.findall(pattern, query_string)
    found_variables = replace_with_center_word(all_variables, synonym_dict)
    new_string = query_string
    # 同时返回替换同义词后的字符串，如何返回？
    for var in all_variables:
        new_string = new_string.replace(var, synonym_dict.get(var, var))

    return found_variables, new_string

def generate_index_dict(inverted_index, variables):
    # 生成索引字典
    index_dict = {}
    for var in variables:
        if var in inverted_index:
            data = inverted_index.get(var, [0])
            index_dict[var] = set(data[1])
        else:
            index_dict[var] = set()
    return index_dict

def evaluate(expr, assignment):
    """评估布尔表达式的值"""
    expr = expr.replace(" AND ", " and ").replace(" OR ", " or ").replace(" NOT ", " not ")
    for var, val in assignment.items():
        expr = expr.replace(var, str(val))
    return eval(expr)

def bool_query_optimized(expr, vars, index_dict):
    # 求index_dict中所有索引集合的并集
    result_set = set()
    
    # 生成所有可能的真值组合
    for assignment in product([False, True], repeat=len(vars)):
        assignment_dict = dict(zip(vars, assignment))
        result = evaluate(expr, assignment_dict)
        temp_set = set.union(*index_dict.values())
        
        # 如果结果为真，则收集该组合的子句
        if result:
            # 判断是否每个变量都为假
            if all(not assignment_dict[v] for v in vars):
                # 报错
                raise ValueError("表达式不合法")
            for var in vars:
                if assignment_dict[var]:  # 如果变量为真，加入原变量
                    # 求temp_set和index_dict交集
                    temp_set = temp_set.intersection(index_dict[var])
            for var in vars:
                if not assignment_dict[var]:  # 如果变量为假，加入NOT原变量
                    # 求temp_set和index_dict相减
                    temp_set = temp_set.difference(index_dict[var])
                
            # 求temp_set和result_set并集
            result_set = result_set.union(temp_set)

    return list(result_set)

def execute_queries(expression, synonym_dict, inverted_index):
    vars, new_expression = extract_variables(expression, synonym_dict)  # 在表达式中出现的变量
    index_dict = generate_index_dict(inverted_index, vars)  # 生成索引字典
    dnf_result = bool_query_optimized(new_expression, vars, index_dict)
    return dnf_result  

# 示例使用
if __name__ == "__main__":
    inverted_index = load_inverted_index('lab1-1/dataset/book_index.json')  # 读取倒排索引
    synonym_dict = load_synonym_dict('lab1-1/dataset/dict_synonym.txt')  # 读取同义词词典
    queries = [
        "(动作 AND 剧情) OR (科幻 AND NOT 恐怖)",
        "((山脉 OR 海洋) AND (夏天 AND NOT 雨天)) OR (城市 AND (夜景 OR 商业))",
        "((学习 AND (编程 OR 数据分析)) AND (工具 OR 资源))",
        "((疫情 OR 疫苗) AND NOT (恐慌 OR 死亡)) OR (健康 AND (生活方式 OR 饮食))",
        "((人工智能 AND NOT (机器人 OR 自动化)) OR (大数据 AND (分析 OR 可视化))) AND (技术 OR 创新)",
        "(小说 AND (爱情 OR 冒险)) OR (非虚构 AND (历史 OR 传记))",
        "((海明威 OR 莎士比亚) AND (小说 OR 诗歌)) AND (经典 OR 现代)",
        "(儿童 AND 图画书 AND (教育 OR 娱乐)) OR (青少年 AND 小说 AND (成长 OR 探险))",
        "((英语 AND (原版 OR 翻译)) AND NOT (外语学习 OR 教材)) OR (文学 AND 科幻)",
        "((悬疑 AND (推理 OR 侦探)) AND NOT (恐怖 OR 血腥)) OR (奇幻 AND (魔法 OR 冒险))",
    ]
    for expression in queries:
        execution_time = timeit.timeit(
            lambda: execute_queries(expression, synonym_dict, inverted_index),
            number=10  # 执行10遍
        )
        dnf_result = execute_queries(expression, synonym_dict, inverted_index)
        print(f"查询语句：{expression}")
        # print(f"查询结果：{dnf_result}")
        print(f"查询耗时：{execution_time:.6f}秒")





